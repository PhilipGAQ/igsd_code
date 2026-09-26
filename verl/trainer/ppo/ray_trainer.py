# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.igsd_ig_compute import parse_trajectory_tokens
from verl.trainer.ppo.igsd_ig_integration import inject_ig_student_into_batch
from verl.trainer.ppo.igsd_m2 import (
    apply_action_likelihood_gate,
    apply_budgeted_local_candidate_target_tilts,
    attach_token_intervention_outputs,
    build_action_logprob_batch,
    build_budgeted_local_candidate_branch_specs,
    build_sampled_action_pair_branch_specs,
    build_sampled_action_positive_mask,
    build_sampled_action_pair_targets,
    build_token_intervention_branch_specs,
    build_student_bc_batch,
    build_teacher_prompt,
    clean_prompt_ids,
    compute_paired_query_ig,
    compute_token_intervention_ig,
    extract_answer_aliases,
    extract_search_queries,
    select_search_turn_indices,
    student_prefix_ids,
)
from verl.trainer.ppo.igsd_teacher_ig import retrieve_sync
from verl.trainer.ppo.igsd_utils import prepare_igsd_batch
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.chat_template import apply_chat_template
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_1d_object_array(values: list[Any]) -> np.ndarray:
    """Preserve nested per-example values as one-dimensional object arrays."""

    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        logger.info(
            "dataloader sizes: train=%s val=%s",
            len(self.train_dataloader),
            len(self.val_dataloader),
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        logger.info("total training steps: %s", self.total_training_steps)

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.warning("failed to set total_training_steps in config: %s", e)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():

            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {}
            for k, v in base_data.items():
                value = v[i]
                if isinstance(value, np.integer):
                    entry[k] = int(value)
                elif isinstance(value, np.floating):
                    entry[k] = float(value)
                else:
                    entry[k] = value
            lines.append(json.dumps(entry, ensure_ascii=False, default=str))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("dumped generations to %s", filename)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _log_rank0(self, message: str, *args, level: int = logging.INFO):
        logger.log(level, message, *args)

    def _merge_validation_rollout_batch(
        self,
        dataset_batch: DataProto,
        rollout_batch: DataProto,
    ) -> DataProto:
        """Merge validation rollout tensors into their dataset carrier.

        Recipe-specific evaluators may override this when the rollout owns an
        authoritative prompt representation. The default path deliberately
        preserves the existing strict ``DataProto.union`` behavior.
        """

        return dataset_batch.union(rollout_batch)

    def _validation_dump_extra_fields(self, batch: DataProto) -> dict[str, Any]:
        """Return recipe-specific row identities for validation JSONL only."""

        data_sources = batch.non_tensor_batch.get("data_source")
        if data_sources is None:
            return {}
        source_values = [data_sources] if isinstance(data_sources, str) else data_sources
        if not any(str(value) == "browsecomp_plus" for value in source_values):
            return {}

        output: dict[str, Any] = {}

        def as_rowwise(value: Any) -> np.ndarray:
            """Normalize a scalar/list/array to one object value per row."""

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return np.full(len(batch), value.item(), dtype=object)
                return value
            if isinstance(value, (list, tuple)):
                return np.asarray(list(value), dtype=object)
            return np.full(len(batch), value, dtype=object)

        def resolve_rowwise_field(key: str) -> Any:
            """Resolve direct or nested agent-loop audit fields."""

            values = batch.non_tensor_batch.get(key)
            if values is not None:
                return as_rowwise(values)
            for carrier_name in ("tool_extra_fields", "extra_fields"):
                carriers = batch.non_tensor_batch.get(carrier_name)
                if carriers is None:
                    continue
                if isinstance(carriers, dict):
                    if key in carriers:
                        return as_rowwise(carriers[key])
                    continue
                if isinstance(carriers, np.ndarray) and carriers.ndim == 0:
                    carriers = [carriers.item()]
                resolved = []
                for carrier in carriers:
                    if isinstance(carrier, dict):
                        resolved.append(carrier.get(key))
                    else:
                        resolved.append(None)
                if resolved:
                    return as_rowwise(resolved)
            return None
        extra_infos = batch.non_tensor_batch.get("extra_info")
        if extra_infos is not None:
            normalized_infos = [value if isinstance(value, dict) else {} for value in extra_infos]

            def normalized_list(value: Any) -> list[Any]:
                if value is None:
                    return []
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [value]

            output["bcp_query_id"] = [str(value.get("bcp_query_id", "")) for value in normalized_infos]
            output["bcp_gold_doc_ids"] = [
                normalized_list(value.get("bcp_gold_doc_ids")) for value in normalized_infos
            ]
            output["bcp_evidence_doc_ids"] = [
                normalized_list(value.get("bcp_evidence_doc_ids")) for value in normalized_infos
            ]

        for key in ("bcp_audit_schema_version", "bcp_retrieved_docids", "bcp_search_calls"):
            values = resolve_rowwise_field(key)
            if values is not None:
                output[key] = values
        return output

    def _validate(self, merged: bool = False):
        self._log_rank0(
            "[driver] validation start: merged=%s batches=%s global_step=%s",
            merged,
            len(self.val_dataloader),
            self.global_steps,
        )
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        validation_dump_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        import tqdm
        for val_batch_idx, test_data in enumerate(tqdm.tqdm(self.val_dataloader), start=1):
            self._log_rank0(
                "[driver] validation batch %s/%s prepare",
                val_batch_idx,
                len(self.val_dataloader),
                level=logging.DEBUG,
            )
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            self._log_rank0(
                "[driver] validation batch %s meta=%s",
                val_batch_idx,
                test_gen_batch.meta_info,
                level=logging.DEBUG,
            )

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            self._log_rank0(
                "[driver] validation batch %s generate start: batch_size=%s padded=%s pad_size=%s",
                val_batch_idx,
                len(test_gen_batch),
                len(test_gen_batch_padded),
                pad_size,
                level=logging.DEBUG,
            )
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            self._log_rank0("[driver] validation batch %s generate done", val_batch_idx, level=logging.DEBUG)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = self._merge_validation_rollout_batch(
                test_batch,
                test_output_gen_batch,
            )
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            dump_extra_fields = self._validation_dump_extra_fields(test_batch)
            for key, values in dump_extra_fields.items():
                if key in reward_extra_infos_dict:
                    raise ValueError(
                        f"Validation dump-only field {key!r} conflicts with reward extra info"
                    )
                if isinstance(values, np.ndarray):
                    normalized_values = values.tolist()
                elif isinstance(values, (list, tuple)):
                    normalized_values = list(values)
                else:
                    normalized_values = [values]
                if len(normalized_values) != len(test_batch):
                    raise ValueError(
                        f"Validation dump-only field {key!r} has {len(normalized_values)} "
                        f"rows for batch size {len(test_batch)}"
                    )
                validation_dump_extra_infos_dict[key].extend(normalized_values)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            dump_extra_infos_dict = dict(reward_extra_infos_dict)
            for key, values in validation_dump_extra_infos_dict.items():
                if key in dump_extra_infos_dict:
                    raise ValueError(
                        f"Validation dump-only field {key!r} conflicts with reward extra info"
                    )
                dump_extra_infos_dict[key] = values
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=dump_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            self._log_rank0("[driver] validation results collected for merge")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        self._log_rank0("[driver] validation done: metric_count=%s", len(metrics))
        return metrics

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._log_rank0("[driver] init_workers: create resource pool start")
        self.resource_pool_manager.create_resource_pool()
        self._log_rank0("[driver] init_workers: create resource pool done")

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            self._log_rank0(
                "[driver] init_workers: spawn worker group pool=%s roles=%s",
                getattr(resource_pool, "name_prefix", type(resource_pool).__name__),
                list(class_dict.keys()),
            )
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self._log_rank0("[driver] init_workers: critic reset start")
            self.critic_wg.reset()
            self._log_rank0("[driver] init_workers: critic reset done")
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self._log_rank0("[driver] init_workers: ref policy init start")
                self.ref_policy_wg.init_model()
                self._log_rank0("[driver] init_workers: ref policy init done")
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self._log_rank0("[driver] init_workers: actor/rollout init start")
        self.actor_rollout_wg.init_model()
        self._log_rank0("[driver] init_workers: actor/rollout init done")

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        self._log_rank0("[driver] init_workers: reward loop manager ready")

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self._log_rank0("[driver] init_workers: async rollout manager create start")
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_model_manager=self.teacher_model_manager,
        )
        self._log_rank0("[driver] init_workers: async rollout manager create done")

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )
        self._log_rank0("[driver] init_workers: checkpoint manager ready")

        # sleep all replicas to load checkpoint
        self._log_rank0("[driver] init_workers: sleep rollout replicas before load checkpoint")
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            compute_loss=False,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")

        old_log_prob_mfu = tu.get(output, "metrics").get("mfu", 0.0)
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        if routed_experts is not None:
            old_log_prob = tu.get_tensordict(
                {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
            )
        else:
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _get_log_prob_size_divisor(self) -> int:
        """Compute the size divisor for padding batches before compute_log_prob.

        The total batch must be divisible by world_size * micro_batch_size_per_gpu:
        - world_size: data is split evenly across workers
        - micro_batch_size_per_gpu: each worker chunks its share into micro-batches
        """
        world_size = max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1)
        mbs = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
        mbs = max(int(mbs or 1), 1)
        return world_size * mbs

    def _compute_old_log_prob_padded(self, batch: DataProto):
        size_divisor = self._get_log_prob_size_divisor()
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch_padded)
        old_log_prob = unpad_dataproto(old_log_prob, pad_size=pad_size)
        return old_log_prob, old_log_prob_mfu

    def _compute_search_turn_entropy_metrics(
        self,
        batch: DataProto,
        entropys: torch.Tensor,
        reward_tensor: torch.Tensor | None = None,
        reward_extra_infos_dict: dict[str, Any] | None = None,
        turn_records: dict[tuple[int, int], dict[str, float | int | None]] | None = None,
    ) -> dict[str, float]:
        if not bool(self.config.algorithm.get("igsd_log_entropy_metrics", False)):
            return {}

        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        reward_extra_infos_dict = reward_extra_infos_dict or {}
        outcome_values: np.ndarray | None = None
        if "origin_score" in reward_extra_infos_dict:
            outcome_values = np.asarray(reward_extra_infos_dict["origin_score"], dtype=np.float32)
        elif reward_tensor is not None:
            outcome_values = reward_tensor.sum(dim=-1).detach().cpu().numpy().astype(np.float32)
        if outcome_values is not None and len(outcome_values) != len(batch):
            outcome_values = None

        sample_rows: list[list[dict[str, float | int | None]]] = [[] for _ in range(len(batch))]
        parse_failure_count = 0
        if turn_records is not None:
            turn_records.clear()
        raw_result_separator = self.config.actor_rollout_ref.rollout.multi_turn.get(
            "summary_result_separator", "\n-*-*-\n"
        )
        result_separator = (
            str(raw_result_separator).replace("\\n", "\n")
            if raw_result_separator
            else "\n-*-*-\n"
        )

        for sample_idx in range(len(batch)):
            try:
                trajectory = parse_trajectory_tokens(
                    response_ids=response_ids[sample_idx],
                    response_mask=response_mask[sample_idx],
                    tokenizer=self.tokenizer,
                    sample_index=sample_idx,
                    result_separator=result_separator,
                )
            except Exception:
                parse_failure_count += 1
                continue

            for turn in trajectory.search_turns:
                start = max(int(turn.query_start), 0)
                end = min(int(turn.query_end), entropys.shape[1])
                if end <= start:
                    continue
                span_mask = response_mask[sample_idx, start:end].bool()
                if not bool(span_mask.any()):
                    continue
                action_entropy = float(
                    entropys[sample_idx, start:end][span_mask].float().mean().detach().cpu().item()
                )
                action_token_count = int(span_mask.sum().detach().cpu().item())
                query_entropy: float | None = None
                non_query_entropy: float | None = None
                query_token_count = 0
                non_query_token_count = 0
                query_spans = list(turn.query_value_spans)
                if not query_spans and turn.query_value_start >= 0:
                    query_spans = [(turn.query_value_start, turn.query_value_end)]
                expected_query_slots = len(turn.query_texts) or int(bool(turn.query_text))
                if expected_query_slots and len(query_spans) == expected_query_slots:
                    local_query_mask = torch.zeros_like(span_mask)
                    spans_valid = True
                    for query_start, query_end in query_spans:
                        query_start = max(int(query_start), start)
                        query_end = min(int(query_end), end)
                        if query_end <= query_start:
                            spans_valid = False
                            break
                        local_query_mask[query_start - start : query_end - start] = True
                    if spans_valid:
                        query_mask = span_mask & local_query_mask
                        if bool(query_mask.any()):
                            query_token_count = int(query_mask.sum().detach().cpu().item())
                            query_entropy = float(
                                entropys[sample_idx, start:end][query_mask]
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                            non_query_mask = span_mask & ~local_query_mask
                            if bool(non_query_mask.any()):
                                non_query_token_count = int(non_query_mask.sum().detach().cpu().item())
                                non_query_entropy = float(
                                    entropys[sample_idx, start:end][non_query_mask]
                                    .float()
                                    .mean()
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                row: dict[str, float | int | None] = {
                    "turn_idx": int(turn.turn_index),
                    "action_entropy": action_entropy,
                    "action_token_count": action_token_count,
                    "query_entropy": query_entropy,
                    "query_token_count": query_token_count,
                    "non_query_entropy": non_query_entropy,
                    "non_query_token_count": non_query_token_count,
                    "outcome": None if outcome_values is None else float(outcome_values[sample_idx]),
                }
                sample_rows[sample_idx].append(row)
                if turn_records is not None:
                    turn_records[(sample_idx, int(turn.turn_index))] = {
                        "action_entropy": action_entropy,
                        "action_token_count": action_token_count,
                        "query_entropy": query_entropy,
                        "query_token_count": query_token_count,
                        "non_query_entropy": non_query_entropy,
                        "non_query_token_count": non_query_token_count,
                    }

        def _mean(values: list[float]) -> float:
            finite = [value for value in values if np.isfinite(value)]
            return float(np.mean(finite)) if finite else 0.0

        def _std(values: list[float]) -> float:
            finite = [value for value in values if np.isfinite(value)]
            return float(np.std(finite)) if finite else 0.0

        def _quantile(values: list[float], q: float) -> float:
            finite = [value for value in values if np.isfinite(value)]
            return float(np.quantile(finite, q)) if finite else 0.0

        def _pearson(left: list[float], right: list[float]) -> float:
            if len(left) < 2 or len(left) != len(right):
                return 0.0
            pairs = [(x, y) for x, y in zip(left, right, strict=True) if np.isfinite(x) and np.isfinite(y)]
            if len(pairs) < 2:
                return 0.0
            left_arr = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
            right_arr = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            if float(left_arr.std()) == 0.0 or float(right_arr.std()) == 0.0:
                return 0.0
            return float(np.corrcoef(left_arr, right_arr)[0, 1])

        def _sample_stat(rows: list[dict[str, float | int | None]], key: str) -> float | None:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            return _mean(values) if values else None

        def _add_success_failure_metrics(prefix: str, values: list[float | None]) -> None:
            if outcome_values is None or len(values) != len(outcome_values):
                return
            success = [
                float(value)
                for value, outcome in zip(values, outcome_values, strict=True)
                if value is not None and outcome > 0
            ]
            failure = [
                float(value)
                for value, outcome in zip(values, outcome_values, strict=True)
                if value is not None and outcome <= 0
            ]
            metrics[f"{prefix}_success_count"] = float(len(success))
            metrics[f"{prefix}_failure_count"] = float(len(failure))
            metrics[f"{prefix}_success_mean"] = _mean(success)
            metrics[f"{prefix}_failure_mean"] = _mean(failure)
            metrics[f"{prefix}_success_failure_gap"] = _mean(success) - _mean(failure)

        all_rows = [row for rows in sample_rows for row in rows]
        action_values = [float(row["action_entropy"]) for row in all_rows]
        action_token_counts = [float(row["action_token_count"]) for row in all_rows]
        query_values = [float(row["query_entropy"]) for row in all_rows if row["query_entropy"] is not None]
        query_token_counts = [
            float(row["query_token_count"]) for row in all_rows if row["query_entropy"] is not None
        ]
        non_query_values = [
            float(row["non_query_entropy"]) for row in all_rows if row["non_query_entropy"] is not None
        ]
        turn_counts = [len(rows) for rows in sample_rows]
        search_sample_count = sum(count > 0 for count in turn_counts)
        sample_action_values = [_sample_stat(rows, "action_entropy") for rows in sample_rows]
        sample_query_values = [_sample_stat(rows, "query_entropy") for rows in sample_rows]
        sample_non_query_values = [_sample_stat(rows, "non_query_entropy") for rows in sample_rows]
        first_action_values = [float(rows[0]["action_entropy"]) if rows else None for rows in sample_rows]
        last_action_values = [float(rows[-1]["action_entropy"]) if rows else None for rows in sample_rows]
        first_query_values = [float(rows[0]["query_entropy"]) if rows and rows[0]["query_entropy"] is not None else None for rows in sample_rows]
        last_query_values = [float(rows[-1]["query_entropy"]) if rows and rows[-1]["query_entropy"] is not None else None for rows in sample_rows]
        action_slopes = [
            (float(rows[-1]["action_entropy"]) - float(rows[0]["action_entropy"])) / (len(rows) - 1)
            if len(rows) >= 2
            else None
            for rows in sample_rows
        ]
        query_slopes: list[float | None] = []
        for rows in sample_rows:
            indexed = [
                (int(row["turn_idx"]), float(row["query_entropy"]))
                for row in rows
                if row["query_entropy"] is not None
            ]
            if len(indexed) >= 2 and indexed[-1][0] > indexed[0][0]:
                query_slopes.append((indexed[-1][1] - indexed[0][1]) / (indexed[-1][0] - indexed[0][0]))
            else:
                query_slopes.append(None)

        action_adjacent_increases: list[float] = []
        query_adjacent_increases: list[float] = []
        action_max_is_last: list[float] = []
        action_max_relative_turn: list[float] = []
        query_max_is_last: list[float] = []
        query_max_relative_turn: list[float] = []
        for rows in sample_rows:
            if not rows:
                continue
            action_seq = [float(row["action_entropy"]) for row in rows]
            action_adjacent_increases.extend(
                float(right > left) for left, right in zip(action_seq[:-1], action_seq[1:], strict=True)
            )
            action_best = int(np.argmax(action_seq))
            action_max_is_last.append(float(action_best == len(rows) - 1))
            action_max_relative_turn.append(action_best / max(len(rows) - 1, 1))

            query_seq = [
                (int(row["turn_idx"]), float(row["query_entropy"]))
                for row in rows
                if row["query_entropy"] is not None
            ]
            query_adjacent_increases.extend(
                float(right[1] > left[1])
                for left, right in zip(query_seq[:-1], query_seq[1:], strict=True)
            )
            if query_seq:
                query_best = int(np.argmax([value for _, value in query_seq]))
                query_best_turn = query_seq[query_best][0]
                query_max_is_last.append(float(query_best_turn == int(rows[-1]["turn_idx"])))
                query_max_relative_turn.append(query_best_turn / max(len(rows) - 1, 1))

        metrics = {
            "igsd_entropy/parse_failure_count": float(parse_failure_count),
            "igsd_entropy/search_sample_count": float(search_sample_count),
            "igsd_entropy/search_sample_frac": float(search_sample_count / max(len(batch), 1)),
            "igsd_entropy/search_action_turn_count": float(len(action_values)),
            "igsd_entropy/search_action_turns_per_sample_mean": _mean([float(v) for v in turn_counts]),
            "igsd_entropy/search_action_turns_per_search_sample_mean": _mean(
                [float(v) for v in turn_counts if v > 0]
            ),
            "igsd_entropy/search_action_turns_per_search_sample_std": _std(
                [float(v) for v in turn_counts if v > 0]
            ),
            "igsd_entropy/search_action_entropy_mean": _mean(action_values),
            "igsd_entropy/search_action_entropy_std": _std(action_values),
            "igsd_entropy/search_action_entropy_p50": _quantile(action_values, 0.5),
            "igsd_entropy/search_action_entropy_p90": _quantile(action_values, 0.9),
            "igsd_entropy/search_action_token_count_mean": _mean(action_token_counts),
            "igsd_entropy/search_action_entropy_token_count_pearson": _pearson(
                action_values, action_token_counts
            ),
            "igsd_entropy/search_action_entropy_first_mean": _mean(
                [float(v) for v in first_action_values if v is not None]
            ),
            "igsd_entropy/search_action_entropy_last_mean": _mean(
                [float(v) for v in last_action_values if v is not None]
            ),
            "igsd_entropy/search_action_entropy_last_minus_first_mean": _mean(
                [
                    float(last) - float(first)
                    for first, last in zip(first_action_values, last_action_values, strict=True)
                    if first is not None and last is not None
                ]
            ),
            "igsd_entropy/search_action_entropy_turn_slope_mean": _mean(
                [float(v) for v in action_slopes if v is not None]
            ),
            "igsd_entropy/search_action_entropy_adjacent_increase_frac": _mean(
                action_adjacent_increases
            ),
            "igsd_entropy/search_action_max_entropy_is_last_frac": _mean(action_max_is_last),
            "igsd_entropy/search_action_max_entropy_relative_turn_mean": _mean(
                action_max_relative_turn
            ),
            "igsd_entropy/query_entropy_turn_count": float(len(query_values)),
            "igsd_entropy/query_entropy_coverage": float(len(query_values) / max(len(action_values), 1)),
            "igsd_entropy/query_entropy_mean": _mean(query_values),
            "igsd_entropy/query_entropy_std": _std(query_values),
            "igsd_entropy/query_entropy_p50": _quantile(query_values, 0.5),
            "igsd_entropy/query_entropy_p90": _quantile(query_values, 0.9),
            "igsd_entropy/query_token_count_mean": _mean(query_token_counts),
            "igsd_entropy/query_entropy_token_count_pearson": _pearson(query_values, query_token_counts),
            "igsd_entropy/query_entropy_first_mean": _mean(
                [float(v) for v in first_query_values if v is not None]
            ),
            "igsd_entropy/query_entropy_last_mean": _mean(
                [float(v) for v in last_query_values if v is not None]
            ),
            "igsd_entropy/query_entropy_last_minus_first_mean": _mean(
                [
                    float(last) - float(first)
                    for first, last in zip(first_query_values, last_query_values, strict=True)
                    if first is not None and last is not None
                ]
            ),
            "igsd_entropy/query_entropy_turn_slope_mean": _mean(
                [float(v) for v in query_slopes if v is not None]
            ),
            "igsd_entropy/query_entropy_adjacent_increase_frac": _mean(query_adjacent_increases),
            "igsd_entropy/query_max_entropy_is_last_frac": _mean(query_max_is_last),
            "igsd_entropy/query_max_entropy_relative_turn_mean": _mean(query_max_relative_turn),
            "igsd_entropy/non_query_entropy_mean": _mean(non_query_values),
            "igsd_entropy/non_query_entropy_std": _std(non_query_values),
            "igsd_entropy/search_action_entropy_turn_count_pearson": _pearson(
                [float(value) for value, count in zip(sample_action_values, turn_counts, strict=True) if value is not None],
                [float(count) for value, count in zip(sample_action_values, turn_counts, strict=True) if value is not None],
            ),
            "igsd_entropy/query_entropy_turn_count_pearson": _pearson(
                [float(value) for value, count in zip(sample_query_values, turn_counts, strict=True) if value is not None],
                [float(count) for value, count in zip(sample_query_values, turn_counts, strict=True) if value is not None],
            ),
        }

        paired_action = [float(row["action_entropy"]) for row in all_rows if row["query_entropy"] is not None]
        paired_query = [float(row["query_entropy"]) for row in all_rows if row["query_entropy"] is not None]
        paired_non_query = [
            float(row["non_query_entropy"])
            for row in all_rows
            if row["query_entropy"] is not None and row["non_query_entropy"] is not None
        ]
        paired_query_for_non_query = [
            float(row["query_entropy"])
            for row in all_rows
            if row["query_entropy"] is not None and row["non_query_entropy"] is not None
        ]
        metrics["igsd_entropy/query_action_entropy_pearson"] = _pearson(paired_query, paired_action)
        metrics["igsd_entropy/query_minus_action_entropy_mean"] = _mean(
            [query - action for query, action in zip(paired_query, paired_action, strict=True)]
        )
        metrics["igsd_entropy/query_non_query_entropy_pearson"] = _pearson(
            paired_query_for_non_query, paired_non_query
        )
        metrics["igsd_entropy/query_minus_non_query_entropy_mean"] = _mean(
            [
                query - non_query
                for query, non_query in zip(paired_query_for_non_query, paired_non_query, strict=True)
            ]
        )

        _add_success_failure_metrics("igsd_entropy/search_action_entropy", sample_action_values)
        _add_success_failure_metrics("igsd_entropy/query_entropy", sample_query_values)
        _add_success_failure_metrics("igsd_entropy/non_query_entropy", sample_non_query_values)
        _add_success_failure_metrics("igsd_entropy/search_action_entropy_first", first_action_values)
        _add_success_failure_metrics("igsd_entropy/search_action_entropy_last", last_action_values)
        _add_success_failure_metrics("igsd_entropy/query_entropy_first", first_query_values)
        _add_success_failure_metrics("igsd_entropy/query_entropy_last", last_query_values)
        _add_success_failure_metrics("igsd_entropy/search_action_entropy_turn_slope", action_slopes)
        _add_success_failure_metrics("igsd_entropy/query_entropy_turn_slope", query_slopes)

        if outcome_values is not None:
            metrics["igsd_entropy/search_action_entropy_outcome_pearson"] = _pearson(
                [float(v) for v in sample_action_values if v is not None],
                [
                    float(outcome)
                    for value, outcome in zip(sample_action_values, outcome_values, strict=True)
                    if value is not None
                ],
            )
            metrics["igsd_entropy/query_entropy_outcome_pearson"] = _pearson(
                [float(v) for v in sample_query_values if v is not None],
                [
                    float(outcome)
                    for value, outcome in zip(sample_query_values, outcome_values, strict=True)
                    if value is not None
                ],
            )
            metrics["igsd_entropy/non_query_entropy_outcome_pearson"] = _pearson(
                [float(v) for v in sample_non_query_values if v is not None],
                [
                    float(outcome)
                    for value, outcome in zip(sample_non_query_values, outcome_values, strict=True)
                    if value is not None
                ],
            )

        max_observed_turns = max(turn_counts, default=0)
        for turn_idx in range(max(6, max_observed_turns)):
            rows = [row for row in all_rows if int(row["turn_idx"]) == turn_idx]
            action_turn_values = [float(row["action_entropy"]) for row in rows]
            query_turn_values = [float(row["query_entropy"]) for row in rows if row["query_entropy"] is not None]
            action_turn_token_counts = [float(row["action_token_count"]) for row in rows]
            query_turn_token_counts = [
                float(row["query_token_count"]) for row in rows if row["query_entropy"] is not None
            ]
            non_query_turn_values = [
                float(row["non_query_entropy"]) for row in rows if row["non_query_entropy"] is not None
            ]
            prefix = f"igsd_entropy/turn{turn_idx + 1}"
            metrics[f"{prefix}_count"] = float(len(rows))
            metrics[f"{prefix}_sample_frac"] = float(len(rows) / max(search_sample_count, 1))
            metrics[f"{prefix}_query_coverage"] = float(len(query_turn_values) / max(len(rows), 1))
            # Preserve the original TensorBoard tag while adding shorter grouped tags.
            metrics[f"igsd_entropy/search_action_entropy_turn{turn_idx + 1}_mean"] = _mean(
                action_turn_values
            )
            metrics[f"{prefix}_search_action_entropy_mean"] = _mean(action_turn_values)
            metrics[f"{prefix}_search_action_token_count_mean"] = _mean(action_turn_token_counts)
            metrics[f"{prefix}_query_entropy_mean"] = _mean(query_turn_values)
            metrics[f"{prefix}_query_token_count_mean"] = _mean(query_turn_token_counts)
            metrics[f"{prefix}_non_query_entropy_mean"] = _mean(non_query_turn_values)
            if outcome_values is not None:
                success_rows = [row for row in rows if row["outcome"] is not None and float(row["outcome"]) > 0]
                failure_rows = [row for row in rows if row["outcome"] is not None and float(row["outcome"]) <= 0]
                success_action = [float(row["action_entropy"]) for row in success_rows]
                failure_action = [float(row["action_entropy"]) for row in failure_rows]
                success_query = [
                    float(row["query_entropy"]) for row in success_rows if row["query_entropy"] is not None
                ]
                failure_query = [
                    float(row["query_entropy"]) for row in failure_rows if row["query_entropy"] is not None
                ]
                success_non_query = [
                    float(row["non_query_entropy"])
                    for row in success_rows
                    if row["non_query_entropy"] is not None
                ]
                failure_non_query = [
                    float(row["non_query_entropy"])
                    for row in failure_rows
                    if row["non_query_entropy"] is not None
                ]
                metrics[f"{prefix}_success_count"] = float(len(success_rows))
                metrics[f"{prefix}_failure_count"] = float(len(failure_rows))
                metrics[f"{prefix}_search_action_entropy_success_mean"] = _mean(success_action)
                metrics[f"{prefix}_search_action_entropy_failure_mean"] = _mean(failure_action)
                metrics[f"{prefix}_search_action_entropy_success_failure_gap"] = _mean(
                    success_action
                ) - _mean(failure_action)
                metrics[f"{prefix}_query_entropy_success_mean"] = _mean(success_query)
                metrics[f"{prefix}_query_entropy_failure_mean"] = _mean(failure_query)
                metrics[f"{prefix}_query_entropy_success_failure_gap"] = _mean(success_query) - _mean(
                    failure_query
                )
                metrics[f"{prefix}_non_query_entropy_success_mean"] = _mean(success_non_query)
                metrics[f"{prefix}_non_query_entropy_failure_mean"] = _mean(failure_non_query)
                metrics[f"{prefix}_non_query_entropy_success_failure_gap"] = _mean(
                    success_non_query
                ) - _mean(failure_non_query)

        for count in range(1, max(6, max_observed_turns) + 1):
            indices = [idx for idx, value in enumerate(turn_counts) if value == count]
            index_set = set(indices)
            prefix = f"igsd_entropy/num_turns{count}"
            metrics[f"{prefix}_sample_count"] = float(len(indices))
            metrics[f"{prefix}_sample_frac"] = float(len(indices) / max(search_sample_count, 1))
            metrics[f"{prefix}_search_action_entropy_mean"] = _mean(
                [float(sample_action_values[idx]) for idx in indices if sample_action_values[idx] is not None]
            )
            metrics[f"{prefix}_query_entropy_mean"] = _mean(
                [float(sample_query_values[idx]) for idx in indices if sample_query_values[idx] is not None]
            )
            metrics[f"{prefix}_non_query_entropy_mean"] = _mean(
                [
                    float(sample_non_query_values[idx])
                    for idx in indices
                    if sample_non_query_values[idx] is not None
                ]
            )
            count_action_values = [
                sample_action_values[idx] if idx in index_set else None for idx in range(len(batch))
            ]
            count_query_values = [
                sample_query_values[idx] if idx in index_set else None for idx in range(len(batch))
            ]
            _add_success_failure_metrics(f"{prefix}_search_action_entropy", count_action_values)
            _add_success_failure_metrics(f"{prefix}_query_entropy", count_query_values)
            metrics[f"{prefix}_last_search_action_entropy_mean"] = _mean(
                [float(last_action_values[idx]) for idx in indices if last_action_values[idx] is not None]
            )
            metrics[f"{prefix}_last_query_entropy_mean"] = _mean(
                [float(last_query_values[idx]) for idx in indices if last_query_values[idx] is not None]
            )
            metrics[f"igsd_entropy/last_is_turn{count}_count"] = float(len(indices))
            metrics[f"igsd_entropy/last_is_turn{count}_frac"] = float(
                len(indices) / max(search_sample_count, 1)
            )

        abnormal_keys = (
            "searched_query_count",
            "too_many_turn_count",
            "too_many_tool_call_count",
            "tool_parser_error_count",
            "response_truncated_count",
            "too_long_seq_truncated_count",
        )
        for key in abnormal_keys:
            values = batch.non_tensor_batch.get(key)
            if values is None or len(values) != len(sample_action_values):
                continue
            arr = np.asarray(values)
            abnormal_action = [
                float(value)
                for value, flag in zip(sample_action_values, arr, strict=True)
                if value is not None and int(flag) > 0
            ]
            normal_action = [
                float(value)
                for value, flag in zip(sample_action_values, arr, strict=True)
                if value is not None and int(flag) <= 0
            ]
            abnormal_query = [
                float(value)
                for value, flag in zip(sample_query_values, arr, strict=True)
                if value is not None and int(flag) > 0
            ]
            normal_query = [
                float(value)
                for value, flag in zip(sample_query_values, arr, strict=True)
                if value is not None and int(flag) <= 0
            ]
            metrics[f"igsd_entropy/search_action_entropy_{key}_mean"] = _mean(abnormal_action)
            metrics[f"igsd_entropy/search_action_entropy_{key}_normal_mean"] = _mean(normal_action)
            metrics[f"igsd_entropy/search_action_entropy_{key}_gap"] = _mean(abnormal_action) - _mean(
                normal_action
            )
            metrics[f"igsd_entropy/query_entropy_{key}_mean"] = _mean(abnormal_query)
            metrics[f"igsd_entropy/query_entropy_{key}_normal_mean"] = _mean(normal_query)
            metrics[f"igsd_entropy/query_entropy_{key}_gap"] = _mean(abnormal_query) - _mean(normal_query)

        return metrics

    def _compute_ref_log_prob_padded(self, batch: DataProto) -> DataProto:
        size_divisor = self._get_log_prob_size_divisor()
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        ref_log_prob = self._compute_ref_log_prob(batch_padded)
        return unpad_dataproto(ref_log_prob, pad_size=pad_size)

    def _igsd_origin_scores(self, batch: DataProto) -> np.ndarray:
        if "origin_score" in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch["origin_score"], dtype=np.float32)
        return np.zeros(len(batch), dtype=np.float32)

    def _igsd_teacher_scores(self, batch: DataProto, origin_scores: np.ndarray) -> np.ndarray:
        if "score" in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch["score"], dtype=np.float32)
        return origin_scores

    def _generate_igsd_token_intervention_branches(
        self,
        teacher_output: DataProto,
    ) -> dict[str, float]:
        """Generate G1 or budgeted local-candidate continuations while replicas are awake."""

        algo_config = self.config.algorithm
        mode = str(algo_config.get("igsd_token_weight_mode", "uniform")).lower()
        if mode not in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }:
            return {"igsd/token_intervention_enabled": 0.0}
        distill_target = str(algo_config.get("igsd_distill_target", "teacher_action")).lower()
        if distill_target != "student_on_policy":
            raise ValueError(
                "algorithm.igsd_token_weight_mode token intervention requires "
                "algorithm.igsd_distill_target='student_on_policy'"
            )
        pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        examples = []
        bypass_prompt_ids = teacher_output.non_tensor_batch.get("igsd_teacher_prompt_ids")
        slot_masks_all = teacher_output.non_tensor_batch.get("igsd_student_action_query_slot_mask")
        queries_all = teacher_output.non_tensor_batch.get("igsd_student_queries")
        query_documents_all = teacher_output.non_tensor_batch.get("igsd_student_query_documents")
        prior_queries_all = teacher_output.non_tensor_batch.get("igsd_prior_queries")
        for idx in range(len(teacher_output)):
            action_ids = list(teacher_output.non_tensor_batch["igsd_student_action_ids"][idx])
            slot_mask = (
                list(slot_masks_all[idx])
                if slot_masks_all is not None
                else [-1] * len(action_ids)
            )
            queries = (
                [str(query) for query in list(queries_all[idx]) if str(query).strip()]
                if queries_all is not None
                else [str(teacher_output.non_tensor_batch["igsd_student_query"][idx])]
            )
            examples.append(
                {
                    "teacher_prompt_ids": list(bypass_prompt_ids[idx])
                    if bypass_prompt_ids is not None
                    else clean_prompt_ids(teacher_output, idx, pad_id),
                    "prefix_ids": list(teacher_output.non_tensor_batch["igsd_prefix_ids"][idx]),
                    "student_action_ids": action_ids,
                    "student_action_query_mask": list(
                        teacher_output.non_tensor_batch["igsd_student_action_query_mask"][idx]
                    ),
                    "student_action_query_slot_mask": slot_mask,
                    "student_query": str(teacher_output.non_tensor_batch["igsd_student_query"][idx]),
                    "student_queries": queries,
                    "student_query_documents": [
                        [str(doc) for doc in documents]
                        for documents in (list(query_documents_all[idx]) if query_documents_all is not None else [])
                    ],
                    "prior_queries": list(prior_queries_all[idx])
                    if prior_queries_all is not None
                    else [],
                }
            )
        default_max_new_tokens = int(
            algo_config.get("igsd_token_intervention_max_new_tokens", 0) or 0
        )
        if default_max_new_tokens <= 0:
            default_max_new_tokens = int(self.config.actor_rollout_ref.rollout.response_length)
        max_model_len = int(self.config.actor_rollout_ref.rollout.max_model_len or 0)
        original_examples = examples
        unaligned_rows = [
            example_idx
            for example_idx, ex in enumerate(original_examples)
            if not any(bool(value) for value in ex.get("student_action_query_mask", []))
        ]
        inactive_indices = set(unaligned_rows)
        overlength_rows: list[tuple[int, int, int]] = []
        if mode == "sampled_action_pair_ig" and max_model_len > 0:
            for example_idx, ex in enumerate(original_examples):
                if example_idx in inactive_indices:
                    continue
                student_length = len(ex["prefix_ids"]) + len(ex["student_action_ids"])
                teacher_length = len(ex["teacher_prompt_ids"]) + len(ex["student_action_ids"])
                if max(student_length, teacher_length) > max_model_len:
                    overlength_rows.append((example_idx, student_length, teacher_length))
            inactive_indices.update(row[0] for row in overlength_rows)
        active_example_indices = [
            idx for idx in range(len(original_examples)) if idx not in inactive_indices
        ]
        examples = [original_examples[idx] for idx in active_example_indices]
        prefilter_metrics: dict[str, float] = {
            "igsd/token_intervention_unaligned_example_count": float(len(unaligned_rows)),
            "igsd/sampled_pair_overlength_example_count": float(len(overlength_rows)),
            "igsd/sampled_pair_overlength_student_count": float(
                sum(student_length > max_model_len for _, student_length, _ in overlength_rows)
            ) if max_model_len > 0 else 0.0,
            "igsd/sampled_pair_overlength_teacher_count": float(
                sum(teacher_length > max_model_len for _, _, teacher_length in overlength_rows)
            ) if max_model_len > 0 else 0.0,
        }
        if not examples:
            teacher_output.non_tensor_batch["igsd_token_branch_records"] = np.empty(
                len(original_examples), dtype=object
            )
            teacher_output.non_tensor_batch["igsd_token_branch_records"][:] = [[] for _ in original_examples]
            teacher_output.non_tensor_batch["igsd_token_budget_plans"] = np.empty(
                len(original_examples), dtype=object
            )
            teacher_output.non_tensor_batch["igsd_token_budget_plans"][:] = [[] for _ in original_examples]
            prefilter_metrics["igsd/token_intervention_enabled"] = 1.0
            prefilter_metrics["igsd/token_intervention_overlength_all_rows"] = float(
                bool(original_examples) and len(overlength_rows) == len(original_examples)
            )
            prefilter_metrics["igsd/token_intervention_unaligned_all_rows"] = float(
                bool(original_examples) and len(unaligned_rows) == len(original_examples)
            )
            return prefilter_metrics
        plans = None
        if mode == "budgeted_local_candidate_ig":
            student_topk_batch = build_action_logprob_batch(
                examples,
                self.tokenizer,
                prompt_key="prefix_ids",
                distill_target="student_on_policy",
            )
            teacher_topk_batch = build_action_logprob_batch(
                examples,
                self.tokenizer,
                prompt_key="teacher_prompt_ids",
                distill_target="student_on_policy",
            )
            if student_topk_batch is None or teacher_topk_batch is None:
                raise ValueError("Budgeted local candidates require aligned student and teacher action batches")
            expected_indices = list(range(len(examples)))
            student_kept = student_topk_batch.non_tensor_batch.get("igsd_kept_example_idx")
            teacher_kept = teacher_topk_batch.non_tensor_batch.get("igsd_kept_example_idx")
            if (
                student_kept is not None
                and [int(index) for index in student_kept.tolist()] != expected_indices
            ) or (
                teacher_kept is not None
                and [int(index) for index in teacher_kept.tolist()] != expected_indices
            ):
                raise RuntimeError(
                    "Budgeted local candidate prefilter refuses to silently reindex teacher/student examples"
                )
            candidate_ids = student_topk_batch.batch["responses"].unsqueeze(-1).long()
            student_query_mask = student_topk_batch.batch.get("igsd_action_query_mask")
            teacher_query_mask = teacher_topk_batch.batch.get("igsd_action_query_mask")
            if student_query_mask is None or teacher_query_mask is None:
                raise RuntimeError("Budgeted local candidate prefilter requires on-policy query masks")
            prefilter_started = time.perf_counter()
            _, student_prefilter_log_probs, student_prefilter_mfu = self._compute_igsd_action_topk(
                student_topk_batch,
                topk=2,
                candidate_ids=candidate_ids,
                response_selection_mask=student_query_mask,
            )
            teacher_prefilter_ids, teacher_prefilter_log_probs, teacher_prefilter_mfu = (
                self._compute_igsd_action_topk(
                    teacher_topk_batch,
                    topk=2,
                    candidate_ids=candidate_ids,
                    response_selection_mask=teacher_query_mask,
                )
            )
            prefilter_seconds = time.perf_counter() - prefilter_started
            if student_prefilter_log_probs is None or teacher_prefilter_log_probs is None:
                raise RuntimeError("Budgeted local candidate prefilter did not return sampled-token log-probs")
            requests, records, plans, planner_metrics = build_budgeted_local_candidate_branch_specs(
                examples,
                teacher_prefilter_ids,
                teacher_prefilter_log_probs,
                student_prefilter_log_probs,
                default_max_new_tokens=default_max_new_tokens,
                max_model_len=max_model_len,
                config=algo_config,
            )
            prefilter_metrics.update(planner_metrics)
            prefilter_metrics.update(
                {
                    "igsd/budgeted_candidate_prefilter_topk": 2.0,
                    "igsd/budgeted_candidate_student_prefilter_mfu": float(student_prefilter_mfu),
                    "igsd/budgeted_candidate_teacher_prefilter_mfu": float(teacher_prefilter_mfu),
                    "timing_s/igsd_budgeted_candidate_prefilter_targets": prefilter_seconds,
                }
            )
        elif mode == "sampled_action_pair_ig":
            student_action_batch = build_action_logprob_batch(
                examples,
                self.tokenizer,
                prompt_key="prefix_ids",
                distill_target="student_on_policy",
            )
            teacher_action_batch = build_action_logprob_batch(
                examples,
                self.tokenizer,
                prompt_key="teacher_prompt_ids",
                distill_target="student_on_policy",
            )
            if student_action_batch is None or teacher_action_batch is None:
                raise ValueError("Sampled-action pair mode requires aligned student and teacher action batches")
            expected_indices = list(range(len(examples)))
            for name, proto in (("student", student_action_batch), ("teacher", teacher_action_batch)):
                kept = proto.non_tensor_batch.get("igsd_kept_example_idx")
                if kept is not None and [int(index) for index in kept.tolist()] != expected_indices:
                    raise RuntimeError(
                        "Sampled-action pair prefilter refuses to silently reindex "
                        f"{name} examples"
                    )
            sampled_ids = student_action_batch.batch["responses"].unsqueeze(-1).long()
            query_mask = student_action_batch.batch.get("igsd_action_query_mask")
            teacher_query_mask = teacher_action_batch.batch.get("igsd_action_query_mask")
            if query_mask is None or teacher_query_mask is None:
                raise RuntimeError("Sampled-action pair prefilter requires on-policy query masks")
            if not torch.equal(query_mask.bool(), teacher_query_mask.bool()):
                raise RuntimeError("Sampled-action pair teacher/student query masks are not aligned")
            prefilter_started = time.perf_counter()
            teacher_prefilter_ids, teacher_log_probs, teacher_mfu = self._compute_igsd_action_topk(
                teacher_action_batch, topk=1, candidate_ids=sampled_ids,
                response_selection_mask=query_mask,
            )
            if teacher_prefilter_ids.shape[-1] < 1:
                raise RuntimeError("Teacher forward returned no top-1 candidate IDs")
            teacher_top1_ids = teacher_prefilter_ids[..., :1]
            # Gather q(t) and q(s) only after the teacher's proposal is known.
            student_candidate_ids = torch.cat([teacher_top1_ids.long(), sampled_ids], dim=-1)
            student_prefilter_ids, student_log_probs, student_mfu = self._compute_igsd_action_topk(
                student_action_batch, topk=1, candidate_ids=student_candidate_ids,
                response_selection_mask=query_mask,
            )
            prefilter_forward_count = 2.0
            if student_prefilter_ids.shape[-1] < 1:
                raise RuntimeError("Sampled-action pair student prefilter returned no top-1 IDs")
            student_top1_ids = student_prefilter_ids[..., :1]
            prefilter_seconds = time.perf_counter() - prefilter_started
            if teacher_log_probs is None or student_log_probs is None:
                raise RuntimeError("Sampled-action pair prefilter did not return candidate log-probs")
            requests, records, plans, planner_metrics = build_sampled_action_pair_branch_specs(
                examples,
                teacher_top1_ids,
                teacher_log_probs,
                student_top1_ids,
                student_log_probs,
                default_max_new_tokens=default_max_new_tokens,
                max_model_len=max_model_len,
                config=algo_config,
                global_step=self.global_steps,
            )
            prefilter_metrics.update(planner_metrics)
            prefilter_metrics.update(
                {
                    "igsd/sampled_pair_prefilter_topk": 1.0,
                    "igsd/sampled_pair_student_prefilter_mfu": float(student_mfu),
                    "igsd/sampled_pair_teacher_prefilter_mfu": float(teacher_mfu),
                    "igsd/sampled_pair_prefilter_forward_count": float(prefilter_forward_count),
                    "timing_s/igsd_sampled_pair_prefilter_targets": prefilter_seconds,
                    "igsd/sampled_pair_active_prefilter_example_count": float(len(examples)),
                }
            )
        else:
            requests, records = build_token_intervention_branch_specs(
                examples,
                default_max_new_tokens=default_max_new_tokens,
                max_model_len=max_model_len,
            )
        max_concurrency = int(algo_config.get("igsd_token_intervention_max_concurrency", 0) or 0)
        rollout_started = time.perf_counter()
        outputs = (
            self.async_rollout_manager.generate_token_continuations(
                requests,
                max_concurrency=max_concurrency,
            )
            if requests
            else []
        )
        rollout_seconds = time.perf_counter() - rollout_started
        parse_metrics = attach_token_intervention_outputs(
            records,
            outputs,
            self.tokenizer,
            retain_branch_action_ids=self._retain_igsd_branch_action_ids(),
            max_queries_per_tool_call=int(
                self.config.actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call or 0
            )
            or None,
            prior_queries_by_example=[list(ex.get("prior_queries", [])) for ex in examples],
        )
        # The auxiliary prefilter may drop unaligned or overlength rows.
        # Reinsert empty records for those rows so downstream OPD packing keeps
        # the original teacher_output order and never silently reindexes targets.
        records_full: list[list[dict[str, Any]]] = [[] for _ in original_examples]
        for local_idx, local_records in enumerate(records):
            original_idx = active_example_indices[local_idx]
            for record in local_records:
                record["example_index"] = original_idx
            records_full[original_idx] = local_records
        records_array = np.empty(len(records_full), dtype=object)
        records_array[:] = records_full
        teacher_output.non_tensor_batch["igsd_token_branch_records"] = records_array
        if plans is not None:
            plans_full: list[list[dict[str, Any]]] = [[] for _ in original_examples]
            for local_idx, local_plans in enumerate(plans):
                plans_full[active_example_indices[local_idx]] = local_plans
            plans_array = np.empty(len(plans_full), dtype=object)
            plans_array[:] = plans_full
            teacher_output.non_tensor_batch["igsd_token_budget_plans"] = plans_array
        parse_metrics.update(
            {
                "igsd/token_intervention_enabled": 1.0,
                "igsd/token_intervention_rollout_example_count": float(len(examples)),
                "igsd/token_intervention_max_new_tokens": float(default_max_new_tokens),
                "igsd/token_intervention_max_concurrency": float(max_concurrency),
                "timing_s/igsd_token_intervention_rollout": rollout_seconds,
            }
        )
        parse_metrics.update(prefilter_metrics)
        return parse_metrics

    def _retain_igsd_branch_action_ids(self) -> bool:
        """Whether branch records must retain raw action tokens after parsing."""

        return False

    def _build_igsd_hindsight_teacher_batch(self, batch: DataProto) -> tuple[DataProto | None, dict[str, float]]:
        algo_config = self.config.algorithm
        provider = str(algo_config.get("igsd_provider", "none"))
        if not bool(algo_config.get("enable_igsd", False)) or provider != "hindsight_rollout":
            return None, {
                "igsd/teacher_provider_active": 0.0,
                "igsd/teacher_aux_count": 0.0,
            }
        context_mode = str(algo_config.get("igsd_teacher_context_mode", "success_queries_score"))
        if context_mode != "success_queries_score":
            raise NotImplementedError(
                "IGSD M2 only supports igsd_teacher_context_mode='success_queries_score' "
                f"to prevent GT or successful-observation leakage, got {context_mode!r}"
            )

        uids = batch.non_tensor_batch.get("uid")
        if uids is None or "raw_prompt" not in batch.non_tensor_batch:
            return None, {
                "igsd/teacher_provider_active": 1.0,
                "igsd/teacher_aux_count": 0.0,
                "igsd/teacher_missing_required_fields": 1.0,
            }

        scores = self._igsd_origin_scores(batch)
        teacher_scores = self._igsd_teacher_scores(batch, scores)
        uid_to_best_success: dict[Any, int] = {}
        for idx, (uid, score) in enumerate(zip(uids, scores, strict=False)):
            if score <= 0.0:
                continue
            prev_idx = uid_to_best_success.get(uid)
            if prev_idx is None or teacher_scores[idx] > teacher_scores[prev_idx]:
                uid_to_best_success[uid] = idx

        per_prompt_limit = int(algo_config.get("igsd_max_teacher_turns_per_prompt", 1) or 1)
        turn_selection = str(algo_config.get("igsd_teacher_turn_selection", "last_search")).lower()
        if turn_selection not in {"last_search", "all_search"}:
            raise NotImplementedError(
                "IGSD M2 supports igsd_teacher_turn_selection='last_search' or 'all_search', "
                f"got {turn_selection!r}"
            )
        pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        raw_result_separator = self.config.actor_rollout_ref.rollout.multi_turn.get(
            "summary_result_separator", "\n-*-*-\n"
        )
        result_separator = (
            str(raw_result_separator).replace("\\n", "\n")
            if raw_result_separator
            else "\n-*-*-\n"
        )
        parsed = [
            parse_trajectory_tokens(
                batch.batch["responses"][idx],
                batch.batch["response_mask"][idx],
                self.tokenizer,
                idx,
                result_separator=result_separator,
            )
            for idx in range(len(batch))
        ]
        strict_query_schema = bool(algo_config.get("igsd_strict_query_schema", False))

        def eligible_turn_indices(sample_idx: int) -> list[int]:
            return [
                turn_idx
                for turn_idx, turn in enumerate(parsed[sample_idx].search_turns)
                if not strict_query_schema or bool(turn.query_schema_valid)
            ]

        invalid_query_schema_turn_count = sum(
            not bool(turn.query_schema_valid)
            for trajectory in parsed
            for turn in trajectory.search_turns
        )
        failed_samples_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, (uid, score) in enumerate(zip(uids, scores, strict=False)):
            success_idx = uid_to_best_success.get(uid)
            if success_idx is None or score > 0.0:
                continue
            if not eligible_turn_indices(idx):
                continue
            failed_samples_by_uid[uid].append(idx)

        selected: list[int] = []
        selected_success: list[int] = []
        selected_turn_indices: list[int] = []
        selected_turn_counts: list[int] = []
        selected_prompt_count = 0
        available_turn_count = 0
        for uid, failed_samples in failed_samples_by_uid.items():
            if turn_selection == "last_search":
                # Preserve the legacy behavior exactly: rank failed siblings by
                # their latest search-turn index and take up to the configured cap.
                candidates = sorted(
                    (
                        (sample_idx, eligible_turn_indices(sample_idx)[-1])
                        for sample_idx in failed_samples
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                limit = len(candidates) if per_prompt_limit <= 0 else per_prompt_limit
                chosen = candidates[:limit]
                available_turn_count += sum(
                    len(eligible_turn_indices(sample_idx)) for sample_idx, _ in chosen
                )
            else:
                # Use one deterministic failed trajectory per prompt so an N-way
                # rollout group does not expand into failures x turns teacher calls.
                sample_idx = max(
                    failed_samples,
                    key=lambda idx: (len(eligible_turn_indices(idx)), -idx),
                )
                available_indices = eligible_turn_indices(sample_idx)
                num_turns = len(available_indices)
                available_turn_count += num_turns
                chosen = [
                    (sample_idx, available_indices[turn_position])
                    for turn_position in select_search_turn_indices(
                        num_turns,
                        mode=turn_selection,
                        limit=per_prompt_limit,
                    )
                ]
            if chosen:
                selected_prompt_count += 1
            for sample_idx, turn_idx in chosen:
                selected.append(sample_idx)
                selected_success.append(uid_to_best_success[uid])
                selected_turn_indices.append(turn_idx)
                selected_turn_counts.append(len(parsed[sample_idx].search_turns))

        if not selected:
            return None, {
                "igsd/teacher_provider_active": 1.0,
                "igsd/teacher_aux_count": 0.0,
                "igsd/teacher_success_prompt_frac": float(len(uid_to_best_success) / max(len(set(uids.tolist())), 1)),
                "igsd/student_invalid_query_schema_turn_count": float(
                    invalid_query_schema_turn_count
                ),
                "igsd/strict_query_schema_enabled": float(strict_query_schema),
            }

        teacher_seed = batch.select_idxs(selected)
        raw_prompts = []
        prefix_ids_all: list[list[int]] = []
        student_queries: list[str] = []
        student_queries_all: list[list[str]] = []
        prior_queries_all: list[list[str]] = []
        student_action_ids_all: list[list[int]] = []
        student_action_query_masks: list[list[int]] = []
        student_action_query_slot_masks: list[list[int]] = []
        student_query_slot_alignment_valid_all: list[bool] = []
        student_query_slot_alignment_expected_all: list[int] = []
        student_query_slot_alignment_actual_all: list[int] = []
        student_documents: list[list[str]] = []
        student_query_documents_all: list[list[list[str]]] = []
        student_tool_response_ids: list[list[int]] = []
        tool_response_template_texts: list[str] = []
        answer_aliases: list[list[str]] = []
        success_queries_all: list[list[str]] = []
        for src_idx, success_idx, turn_idx in zip(selected, selected_success, selected_turn_indices, strict=True):
            turn = parsed[src_idx].search_turns[turn_idx]
            prefix_ids = student_prefix_ids(batch, src_idx, turn, pad_id)
            # The original question is already present in raw_prompt. Include only
            # the generated failed trace here to avoid duplicating the question.
            prefix_text = self.tokenizer.decode(
                batch.batch["responses"][src_idx, : turn.query_start], skip_special_tokens=False
            )
            success_queries = [
                query
                for success_turn in parsed[success_idx].search_turns
                if not strict_query_schema or bool(success_turn.query_schema_valid)
                for query in (
                    success_turn.query_texts
                    or ([success_turn.query_text] if success_turn.query_text else [])
                )
            ]
            raw_prompts.append(
                build_teacher_prompt(
                    raw_prompt=batch.non_tensor_batch["raw_prompt"][src_idx],
                    failed_prefix_text=prefix_text[-12000:],
                    successful_queries=success_queries,
                    success_score=float(teacher_scores[success_idx]),
                    max_queries_per_tool_call=int(
                        self.config.actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call or 1
                    ),
                )
            )
            prefix_ids_all.append(prefix_ids)
            student_queries.append(turn.query_text)
            turn_queries = list(turn.query_texts or ([turn.query_text] if turn.query_text else []))
            student_queries_all.append(turn_queries)
            prior_queries_all.append(
                [
                    query
                    for prior_turn in parsed[src_idx].search_turns[:turn_idx]
                    if not strict_query_schema or bool(prior_turn.query_schema_valid)
                    for query in (
                        prior_turn.query_texts
                        or ([prior_turn.query_text] if prior_turn.query_text else [])
                    )
                ]
            )
            student_action_ids = batch.batch["responses"][src_idx, turn.query_start : turn.query_end].tolist()
            student_action_query_mask = [0] * len(student_action_ids)
            student_action_query_slot_mask = [-1] * len(student_action_ids)
            query_spans = list(turn.query_value_spans)
            if not query_spans and (
                turn.query_value_start >= turn.query_start
                and turn.query_value_end > turn.query_value_start
                and turn.query_value_end <= turn.query_end
            ):
                query_spans = [(turn.query_value_start, turn.query_value_end)]
            expected_query_slots = len(turn_queries)
            actual_query_slots = len(query_spans)
            query_slot_alignment_valid = bool(expected_query_slots) and (
                actual_query_slots == expected_query_slots
            )
            previous_span_end = turn.query_start
            for span_start, span_end in query_spans:
                if (
                    span_start < turn.query_start
                    or span_end > turn.query_end
                    or span_end <= span_start
                    or span_start < previous_span_end
                ):
                    query_slot_alignment_valid = False
                    break
                previous_span_end = span_end
            # Multi-query token intervention must never silently supervise only
            # the slots whose decode-to-token mapping happened to succeed.  An
            # all-zero mask makes the complete auxiliary row fail closed while
            # leaving the original RL sample untouched.
            if not query_slot_alignment_valid:
                query_spans = []
            for slot_idx, (span_start, span_end) in enumerate(query_spans):
                if span_start < turn.query_start or span_end > turn.query_end or span_end <= span_start:
                    continue
                local_start = span_start - turn.query_start
                local_end = span_end - turn.query_start
                student_action_query_mask[local_start:local_end] = [1] * (local_end - local_start)
                student_action_query_slot_mask[local_start:local_end] = [slot_idx] * (local_end - local_start)
            student_action_ids_all.append(student_action_ids)
            student_action_query_masks.append(student_action_query_mask)
            student_action_query_slot_masks.append(student_action_query_slot_mask)
            student_query_slot_alignment_valid_all.append(query_slot_alignment_valid)
            student_query_slot_alignment_expected_all.append(expected_query_slots)
            student_query_slot_alignment_actual_all.append(actual_query_slots)
            summary_enabled = bool(
                self.config.actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary
            )
            student_query_documents = (
                []
                if summary_enabled
                else [list(documents) for documents in turn.documents_by_query]
            )
            if not student_query_documents and not summary_enabled and turn.documents:
                student_query_documents = [list(turn.documents)]
            student_query_documents_all.append(student_query_documents)
            student_documents.append(
                [doc for documents in student_query_documents for doc in documents]
                if student_query_documents
                else ([] if summary_enabled else list(turn.documents))
            )
            tool_ids = batch.batch[
                "responses"
            ][src_idx, turn.tool_response_start : turn.tool_response_end].tolist()
            while tool_ids and tool_ids[-1] == pad_id:
                tool_ids.pop()
            student_tool_response_ids.append(tool_ids)
            tool_response_template_texts.append(
                self.tokenizer.decode(tool_ids, skip_special_tokens=False)
            )
            answer_aliases.append(
                extract_answer_aliases(
                    batch, src_idx, int(algo_config.get("igsd_max_gt_aliases", 3) or 3)
                )
            )
            success_queries_all.append(success_queries)
        teacher_seed.non_tensor_batch["raw_prompt"] = np.array(raw_prompts, dtype=object)
        teacher_seed.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(teacher_seed), dtype=object)

        row_verification_mode = str(algo_config.get("igsd_row_verification_mode", "paired")).lower()
        token_intervention_metrics: dict[str, float] = {}

        def attach_teacher_metadata(output: DataProto) -> None:
            # Explicit mapping is required because the teacher prompt is privileged,
            # while the OPD batch must be rebuilt from the original student prefix.
            output.non_tensor_batch["igsd_source_sample_idx"] = np.array(selected, dtype=object)
            output.non_tensor_batch["igsd_source_turn_idx"] = np.array(selected_turn_indices, dtype=object)
            output.non_tensor_batch["igsd_source_turn_count"] = np.array(selected_turn_counts, dtype=object)
            output.non_tensor_batch["igsd_prefix_ids"] = _to_1d_object_array(prefix_ids_all)
            output.non_tensor_batch["igsd_student_query"] = np.array(student_queries, dtype=object)
            output.non_tensor_batch["igsd_student_queries"] = _to_1d_object_array(student_queries_all)
            output.non_tensor_batch["igsd_student_query_count"] = np.array(
                [len(queries) for queries in student_queries_all], dtype=object
            )
            output.non_tensor_batch["igsd_prior_queries"] = _to_1d_object_array(prior_queries_all)
            output.non_tensor_batch["igsd_student_action_ids"] = _to_1d_object_array(student_action_ids_all)
            output.non_tensor_batch["igsd_student_action_query_mask"] = _to_1d_object_array(
                student_action_query_masks
            )
            output.non_tensor_batch["igsd_student_action_query_slot_mask"] = _to_1d_object_array(
                student_action_query_slot_masks
            )
            output.non_tensor_batch["igsd_student_query_slot_alignment_valid"] = np.array(
                student_query_slot_alignment_valid_all, dtype=object
            )
            output.non_tensor_batch["igsd_student_query_slot_alignment_expected"] = np.array(
                student_query_slot_alignment_expected_all, dtype=object
            )
            output.non_tensor_batch["igsd_student_query_slot_alignment_actual"] = np.array(
                student_query_slot_alignment_actual_all, dtype=object
            )
            output.non_tensor_batch["igsd_student_documents"] = _to_1d_object_array(student_documents)
            output.non_tensor_batch["igsd_student_query_documents"] = _to_1d_object_array(
                student_query_documents_all
            )
            output.non_tensor_batch["igsd_student_tool_response_ids"] = _to_1d_object_array(
                student_tool_response_ids
            )
            output.non_tensor_batch["igsd_tool_response_template_text"] = np.array(
                tool_response_template_texts, dtype=object
            )
            output.non_tensor_batch["igsd_answer_aliases"] = _to_1d_object_array(answer_aliases)
            output.non_tensor_batch["igsd_success_queries"] = _to_1d_object_array(success_queries_all)

        if row_verification_mode == "bypass":
            # Tokenize the privileged prompt locally with the same chat-template
            # settings as the agent loop; the original carrier prompts are still
            # the student prompts and must not be reused here.
            apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
            teacher_prompt_ids_all: list[list[int]] = []
            for messages in raw_prompts:
                if self.processor is not None:
                    rendered = apply_chat_template(
                        self.processor,
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **apply_kwargs,
                    )
                    model_inputs = self.processor(text=[rendered], return_tensors="pt")
                    prompt_ids = normalize_token_ids(model_inputs["input_ids"])
                else:
                    prompt_ids = normalize_token_ids(
                        apply_chat_template(
                            self.tokenizer,
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            **apply_kwargs,
                        )
                    )
                teacher_prompt_ids_all.append(prompt_ids)
            teacher_seed.non_tensor_batch["igsd_teacher_prompt_ids"] = _to_1d_object_array(
                teacher_prompt_ids_all
            )
            attach_teacher_metadata(teacher_seed)
            teacher_output = teacher_seed
            if str(algo_config.get("igsd_token_weight_mode", "uniform")).lower() in {
                "prefix_intervention_ig",
                "budgeted_local_candidate_ig",
                "sampled_action_pair_ig",
            }:
                self.checkpoint_manager.update_weights(self.global_steps)
                try:
                    token_intervention_metrics = self._generate_igsd_token_intervention_branches(teacher_output)
                finally:
                    self.checkpoint_manager.sleep_replicas()
            else:
                token_intervention_metrics = {"igsd/token_intervention_enabled": 0.0}
        else:
            teacher_gen_batch = self._get_gen_batch(teacher_seed)
            teacher_gen_batch.meta_info["global_steps"] = self.global_steps
            size_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
            teacher_gen_batch_padded, pad_size = pad_dataproto_to_divisor(teacher_gen_batch, size_divisor)
            self.checkpoint_manager.update_weights(self.global_steps)
            try:
                teacher_output_padded = self.async_rollout_manager.generate_sequences(teacher_gen_batch_padded)
                teacher_output_padded.meta_info.pop("timing", None)
                teacher_output = unpad_dataproto(teacher_output_padded, pad_size=pad_size)
                if "response_mask" not in teacher_output.batch.keys():
                    teacher_output.batch["response_mask"] = compute_response_mask(teacher_output)
                attach_teacher_metadata(teacher_output)
                if teacher_output.batch["response_mask"].sum() > 0:
                    token_intervention_metrics = self._generate_igsd_token_intervention_branches(teacher_output)
                else:
                    token_intervention_metrics = {
                        "igsd/token_intervention_enabled": float(
                            str(algo_config.get("igsd_token_weight_mode", "uniform")).lower()
                            in {
                                "prefix_intervention_ig",
                                "budgeted_local_candidate_ig",
                                "sampled_action_pair_ig",
                            }
                        )
                    }
            finally:
                self.checkpoint_manager.sleep_replicas()

            # Safety: if all teacher responses are empty (no generated tokens), skip this batch.
            teacher_resp_mask = teacher_output.batch["response_mask"]
            if teacher_resp_mask.sum() == 0:
                self._log_rank0(
                    "[driver] step %s IGSD: teacher rollout produced zero response tokens, skipping",
                    self.global_steps,
                    level=logging.WARNING,
                )
                empty_metrics = {
                    "igsd/teacher_provider_active": 1.0,
                    "igsd/teacher_aux_count": 0.0,
                    "igsd/teacher_empty_responses": 1.0,
                    "igsd/teacher_success_prompt_frac": float(
                        len(uid_to_best_success) / max(len(set(uids.tolist())), 1)
                    ),
                }
                empty_metrics.update(token_intervention_metrics)
                return None, empty_metrics

        metrics = {
            "igsd/teacher_provider_active": 1.0,
            "igsd/teacher_aux_count": float(len(teacher_output)),
            "igsd/teacher_aux_frac": float(len(teacher_output) / max(len(batch), 1)),
            "igsd/teacher_success_prompt_frac": float(len(uid_to_best_success) / max(len(set(uids.tolist())), 1)),
            "igsd/teacher_selected_turn_mean": float(np.mean(selected_turn_indices)),
            "igsd/teacher_selected_turn_min": float(min(selected_turn_indices)),
            "igsd/teacher_selected_turn_max": float(max(selected_turn_indices)),
            "igsd/teacher_selected_prompt_count": float(selected_prompt_count),
            "igsd/teacher_selected_turn_count": float(len(selected_turn_indices)),
            "igsd/teacher_selected_turns_per_prompt_mean": float(
                len(selected_turn_indices) / max(selected_prompt_count, 1)
            ),
            "igsd/teacher_available_turn_count": float(available_turn_count),
            "igsd/teacher_selected_first_turn_frac": float(
                np.mean([turn_idx == 0 for turn_idx in selected_turn_indices])
            ),
            "igsd/teacher_selected_last_turn_frac": float(
                np.mean(
                    [
                        turn_idx >= max(turn_count - 1, 0)
                        for turn_idx, turn_count in zip(selected_turn_indices, selected_turn_counts, strict=True)
                    ]
                )
            ),
            "igsd/teacher_selected_relative_turn_mean": float(
                np.mean(
                    [
                        turn_idx / max(turn_count - 1, 1)
                        for turn_idx, turn_count in zip(selected_turn_indices, selected_turn_counts, strict=True)
                    ]
                )
            ),
            "igsd/student_invalid_query_schema_turn_count": float(
                invalid_query_schema_turn_count
            ),
            "igsd/strict_query_schema_enabled": float(strict_query_schema),
        }
        metrics.update(token_intervention_metrics)
        for turn_idx in sorted(set(selected_turn_indices)):
            metrics[f"igsd/teacher_selected_turn_{turn_idx}_count"] = float(
                sum(value == turn_idx for value in selected_turn_indices)
            )
        return teacher_output, metrics

    def _compute_igsd_action_topk(
        self,
        batch: DataProto,
        *,
        topk: int,
        candidate_ids: torch.Tensor | None = None,
        response_selection_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        """Return action-position top-k IDs and optional candidate log-probs.

        ``response_selection_mask`` restricts expensive full-vocabulary target
        materialization while preserving the returned action-position alignment.
        It is intentionally optional so legacy full-action teacher targets are
        unchanged.
        """

        if topk <= 0:
            raise ValueError(f"algorithm.igsd_distill_topk must be positive, got {topk}")
        if candidate_ids is not None:
            if candidate_ids.shape[:2] != batch.batch["responses"].shape:
                raise ValueError(
                    "IGSD candidate IDs must align with the canonical action response: "
                    f"{candidate_ids.shape[:2]=}, {batch.batch['responses'].shape=}"
                )
            batch.batch["igsd_candidate_ids"] = candidate_ids.long()
        if response_selection_mask is not None:
            if response_selection_mask.shape != batch.batch["responses"].shape:
                raise ValueError(
                    "IGSD response selection mask must align with the canonical action response: "
                    f"{response_selection_mask.shape=}, {batch.batch['responses'].shape=}"
                )
            batch.batch["igsd_return_topk_mask"] = response_selection_mask.bool()

        size_divisor = self._get_log_prob_size_divisor()
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        batch_td = left_right_2_no_padding(batch_padded.to_tensordict())
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            compute_loss=False,
            igsd_return_topk=topk,
            # The prefilter consumes only compact top-k IDs/log-probs.  Do not
            # also materialize ordinary per-token log_probs for this internal
            # forward; PPO/ref and actor-update paths keep their old behavior.
            igsd_topk_only=True,
            igsd_topk_materialization_chunk_size=int(
                self.config.algorithm.get("igsd_topk_materialization_chunk_size", 512) or 0
            ),
            igsd_prefilter_selective_lm_head=bool(
                self.config.algorithm.get("igsd_prefilter_selective_lm_head", False)
            ),
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        topk_ids = tu.get(output, "igsd_topk_ids").to_padded_tensor(0).long()
        topk_log_probs = None
        if candidate_ids is not None:
            topk_log_probs = tu.get(output, "igsd_topk_log_probs").to_padded_tensor(0.0).float()
        if pad_size:
            topk_ids = topk_ids[:-pad_size]
            if topk_log_probs is not None:
                topk_log_probs = topk_log_probs[:-pad_size]
        if topk_ids.shape[0] != len(batch):
            raise AssertionError(f"IGSD top-k output batch size {topk_ids.shape[0]} does not match input {len(batch)}")
        mfu = float(tu.get(output, "metrics").get("mfu", 0.0))
        return topk_ids, topk_log_probs, mfu

    def _build_igsd_m2_bc_batch(
        self,
        teacher_output: DataProto,
        turn_entropy_records: dict[tuple[int, int], dict[str, float | int | None]] | None = None,
    ) -> tuple[DataProto | None, dict[str, float]]:
        """Build paired-IG-gated teacher-action or on-policy distillation examples."""

        distill_mode = str(self.config.algorithm.get("igsd_distill_mode", "bc")).lower()
        distill_target = str(self.config.algorithm.get("igsd_distill_target", "teacher_action")).lower()
        raw_result_separator = self.config.actor_rollout_ref.rollout.multi_turn.get(
            "summary_result_separator", "\n-*-*-\n"
        )
        result_separator = (
            str(raw_result_separator).replace("\\n", "\n")
            if raw_result_separator
            else "\n-*-*-\n"
        )
        row_verification_mode = str(
            self.config.algorithm.get("igsd_row_verification_mode", "paired")
        ).lower()
        supported_distill_modes = {
            "bc",
            "nll",
            "jsd",
            "event_jsd",
            "forward_kl",
            "reverse_kl",
            "event_reverse_kl",
            "topk_reverse_kl",
            "topk_jsd",
            "candidate_pair_jsd",
            "confidence_bc",
        }
        if distill_mode not in supported_distill_modes:
            raise NotImplementedError(
                "IGSD M2 supports distill modes "
                f"{sorted(supported_distill_modes)}, got {distill_mode!r}"
            )
        supported_distill_targets = {"teacher_action", "student_on_policy"}
        if distill_target not in supported_distill_targets:
            raise NotImplementedError(
                "IGSD M2 supports distill targets "
                f"{sorted(supported_distill_targets)}, got {distill_target!r}"
            )
        distill_span = str(self.config.algorithm.get("igsd_distill_span", "full_tool_call")).lower()
        supported_distill_spans = {"full_tool_call", "query_only"}
        if distill_span not in supported_distill_spans:
            raise NotImplementedError(
                "IGSD M2 supports distill spans "
                f"{sorted(supported_distill_spans)}, got {distill_span!r}"
            )
        token_weight_mode = str(self.config.algorithm.get("igsd_token_weight_mode", "uniform")).lower()
        sampled_pair_gate_source = str(
            self.config.algorithm.get("igsd_sampled_pair_gate_source", "environment_ig")
        ).lower()
        sampled_pair_gate_granularity = str(
            self.config.algorithm.get("igsd_sampled_pair_gate_granularity", "token")
        ).lower()
        preserve_zero_token_gate_rows = bool(
            self.config.algorithm.get("igsd_preserve_zero_token_gate_rows", False)
        )
        if token_weight_mode not in {
            "uniform",
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }:
            raise NotImplementedError(
                "IGSD M2 supports token weighting modes ['uniform', 'prefix_intervention_ig', "
                "'budgeted_local_candidate_ig', 'sampled_action_pair_ig'], "
                f"got {token_weight_mode!r}"
            )
        if (
            not bool(self.config.algorithm.get("igsd_enable_ig_forward", False))
            and row_verification_mode != "bypass"
        ):
            raise NotImplementedError(
                "IGSD M2 requires algorithm.igsd_enable_ig_forward=True unless "
                "row verification mode is bypass"
            )
        if (
            not bool(self.config.algorithm.get("igsd_enable_ig_forward", False))
            and (
                token_weight_mode in {"prefix_intervention_ig", "budgeted_local_candidate_ig"}
                or (
                    token_weight_mode == "sampled_action_pair_ig"
                    and sampled_pair_gate_source == "environment_ig"
                )
            )
        ):
            raise NotImplementedError(
                "Environment-verified token weighting requires algorithm.igsd_enable_ig_forward=True"
            )
        if token_weight_mode in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        } and distill_target != "student_on_policy":
            raise ValueError(
                "algorithm.igsd_token_weight_mode token intervention requires "
                "algorithm.igsd_distill_target='student_on_policy'"
            )
        topk = int(self.config.algorithm.get("igsd_distill_topk", 50) or 50)
        if topk <= 0:
            raise ValueError(f"algorithm.igsd_distill_topk must be positive, got {topk}")
        materialization_chunk_size = int(
            self.config.algorithm.get("igsd_topk_materialization_chunk_size", 512)
        )
        if materialization_chunk_size < 0:
            raise ValueError(
                "algorithm.igsd_topk_materialization_chunk_size must be non-negative, "
                f"got {materialization_chunk_size}"
            )
        if distill_mode == "candidate_pair_jsd" and topk != 2:
            raise ValueError(
                "algorithm.igsd_distill_topk must equal 2 for candidate_pair_jsd; "
                "the candidate-pair objective has a fixed two-token support"
            )
        floor_log_prob = float(self.config.algorithm.get("igsd_rkl_floor_log_prob", -30.0))
        if floor_log_prob > 0.0:
            raise ValueError(f"algorithm.igsd_rkl_floor_log_prob must be <= 0, got {floor_log_prob}")
        if distill_mode in {"topk_reverse_kl", "topk_jsd", "candidate_pair_jsd"}:
            actor_config = self.config.actor_rollout_ref.actor
            model_config = self.config.actor_rollout_ref.model
            if is_distillation_enabled(self.config.get("distillation")):
                raise ValueError("IGSD top-k distillation cannot be combined with generic distillation.enabled=True")
            if str(actor_config.get("strategy", "fsdp")) not in {"fsdp", "fsdp2"}:
                raise NotImplementedError("IGSD top-k distillation currently supports FSDP/FSDP2 actors only")
            if not bool(model_config.get("use_remove_padding", False)):
                raise ValueError("IGSD top-k distillation requires actor_rollout_ref.model.use_remove_padding=True")
            if bool(model_config.get("use_fused_kernels", False)):
                raise ValueError("IGSD top-k distillation requires actor_rollout_ref.model.use_fused_kernels=False")
            if int(actor_config.get("ulysses_sequence_parallel_size", 1)) != 1:
                raise ValueError("IGSD top-k distillation requires actor sequence parallel size 1")

        if token_weight_mode == "sampled_action_pair_ig":
            if sampled_pair_gate_granularity not in {"token", "query_mean"}:
                raise ValueError(
                    "igsd_sampled_pair_gate_granularity must be 'token' or 'query_mean', "
                    f"got {sampled_pair_gate_granularity!r}"
                )
            if (
                sampled_pair_gate_granularity == "query_mean"
                and sampled_pair_gate_source != "environment_ig"
            ):
                raise ValueError(
                    "igsd_sampled_pair_gate_granularity='query_mean' requires "
                    "igsd_sampled_pair_gate_source='environment_ig'"
                )
            if sampled_pair_gate_source not in {
                "environment_ig",
                "likelihood_gap",
                "constant",
            }:
                raise ValueError(
                    "igsd_sampled_pair_gate_source must be 'environment_ig', "
                    "'likelihood_gap', or 'constant', "
                    f"got {sampled_pair_gate_source!r}"
                )
            if distill_mode not in {"candidate_pair_jsd", "topk_jsd", "topk_reverse_kl"}:
                raise ValueError(
                    "sampled_action_pair_ig requires igsd_distill_mode='candidate_pair_jsd', "
                    "'topk_jsd', or 'topk_reverse_kl'"
                )
            sampled_pair_reference_mode = str(
                self.config.algorithm.get("igsd_sampled_pair_reference_mode", "greedy_pair")
            ).lower()
            if sampled_pair_reference_mode not in {"greedy_pair", "fixed_s"}:
                raise ValueError(
                    "igsd_sampled_pair_reference_mode must be 'greedy_pair' or 'fixed_s', "
                    f"got {sampled_pair_reference_mode!r}"
                )
            sampled_pair_candidate_mode = str(
                self.config.algorithm.get("igsd_sampled_pair_candidate_mode", "sampled_action")
            ).lower()
            if sampled_pair_candidate_mode not in {"sampled_action", "student_top1"}:
                raise ValueError(
                    "igsd_sampled_pair_candidate_mode must be 'sampled_action' or 'student_top1', "
                    f"got {sampled_pair_candidate_mode!r}"
                )
            if (
                sampled_pair_candidate_mode == "student_top1"
                and sampled_pair_reference_mode != "greedy_pair"
            ):
                raise ValueError(
                    "igsd_sampled_pair_candidate_mode='student_top1' requires "
                    "igsd_sampled_pair_reference_mode='greedy_pair'"
                )
            if (
                sampled_pair_gate_source == "likelihood_gap"
                and sampled_pair_candidate_mode != "sampled_action"
            ):
                raise ValueError(
                    "igsd_sampled_pair_gate_source='likelihood_gap' requires "
                    "igsd_sampled_pair_candidate_mode='sampled_action'"
                )
            if distill_target != "student_on_policy" or distill_span != "query_only":
                raise ValueError(
                    "sampled_action_pair_ig requires student_on_policy/query_only distillation"
                )
            if row_verification_mode != "bypass":
                raise ValueError("sampled_action_pair_ig requires igsd_row_verification_mode='bypass'")
            if str(self.config.algorithm.get("igsd_gate_mode", "sigmoid")).lower() != "none":
                raise ValueError("sampled_action_pair_ig requires igsd_gate_mode='none'")
            if str(self.config.algorithm.get("igsd_token_gate_mode", "sigmoid")).lower() not in {
                "rectified_sigmoid",
                "hard",
            }:
                raise ValueError(
                    "sampled_action_pair_ig requires igsd_token_gate_mode='rectified_sigmoid' or 'hard'"
                )
            if str(self.config.algorithm.get("igsd_token_gate_normalization", "row_mean")).lower() != "none":
                raise ValueError("sampled_action_pair_ig requires igsd_token_gate_normalization='none'")
            if float(self.config.algorithm.get("igsd_token_gate_margin", 0.0)) < 0.0:
                raise ValueError(
                    "sampled_action_pair_ig requires igsd_token_gate_margin >= 0 for positive-only pair gating"
                )
            sampled_pair_budget_ratio = float(
                self.config.algorithm.get("igsd_sampled_pair_budget_ratio", 1.0)
            )
            sampled_pair_audit_all_eligible = bool(
                self.config.algorithm.get("igsd_sampled_pair_audit_all_eligible", False)
            )
            sampled_pair_max_budget_ratio = 2.0 if sampled_pair_audit_all_eligible else 1.0
            if (
                not math.isfinite(sampled_pair_budget_ratio)
                or not 0.0 <= sampled_pair_budget_ratio <= sampled_pair_max_budget_ratio
            ):
                raise ValueError(
                    "sampled_action_pair_ig requires igsd_sampled_pair_budget_ratio in "
                    f"[0, {sampled_pair_max_budget_ratio:g}] when "
                    "igsd_sampled_pair_audit_all_eligible="
                    f"{sampled_pair_audit_all_eligible}"
                )
            sampled_pair_log_odds_epsilon = float(
                self.config.algorithm.get("igsd_sampled_pair_log_odds_epsilon", 1e-6)
            )
            if not math.isfinite(sampled_pair_log_odds_epsilon) or sampled_pair_log_odds_epsilon < 0.0:
                raise ValueError(
                    "sampled_action_pair_ig requires a finite non-negative "
                    "igsd_sampled_pair_log_odds_epsilon"
                )
            if float(self.config.algorithm.get("igsd_token_invalid_fallback_weight", 0.0)) != 0.0:
                raise ValueError("sampled_action_pair_ig requires igsd_token_invalid_fallback_weight=0")
            if not preserve_zero_token_gate_rows:
                raise ValueError("sampled_action_pair_ig requires igsd_preserve_zero_token_gate_rows=True")
        elif distill_mode == "candidate_pair_jsd":
            raise ValueError(
                "candidate_pair_jsd requires igsd_token_weight_mode='sampled_action_pair_ig'"
            )

        row_verification_mode = str(
            self.config.algorithm.get("igsd_row_verification_mode", "paired")
        ).lower()
        bypass_row_verification = row_verification_mode == "bypass"
        if bypass_row_verification:
            teacher_queries = ["" for _ in range(len(teacher_output))]
            valid_indices = list(range(len(teacher_output)))
        else:
            decoded = [
                self.tokenizer.decode(
                    teacher_output.batch["responses"][idx][teacher_output.batch["response_mask"][idx].bool()],
                    skip_special_tokens=False,
                )
                for idx in range(len(teacher_output))
            ]
            teacher_query_lists = [
                extract_search_queries(
                    text,
                    strict_schema=bool(
                        self.config.algorithm.get("igsd_strict_query_schema", False)
                    ),
                )
                for text in decoded
            ]
            teacher_queries = [" ".join(queries) for queries in teacher_query_lists]
            valid_indices = [idx for idx, queries in enumerate(teacher_query_lists) if queries]
        if bypass_row_verification:
            teacher_query_lists = [[] for _ in range(len(teacher_output))]
        teacher_valid_query_count = sum(bool(queries) for queries in teacher_query_lists)
        bypass_candidate_count = len(valid_indices) if bypass_row_verification else 0
        source_turn_indices = [
            int(value) for value in teacher_output.non_tensor_batch["igsd_source_turn_idx"].tolist()
        ]
        routing_mode = str(self.config.algorithm.get("igsd_turn_routing_mode", "independent")).lower()
        metrics = {
            "igsd/teacher_valid_query_count": float(teacher_valid_query_count),
            "igsd/teacher_valid_query_frac": float(
                teacher_valid_query_count / max(len(teacher_output), 1)
            ),
            # In bypass mode ``valid_indices`` are OPD carriers, not parsed
            # teacher replacement queries. Keep the two concepts separate.
            "igsd/bypass_candidate_count": float(bypass_candidate_count),
            "igsd/bypass_candidate_frac": float(
                bypass_candidate_count / max(len(teacher_output), 1)
            ),
            "igsd/m2_turn_routing_is_independent": float(routing_mode == "independent"),
            "igsd/m2_turn_routing_is_prompt_normalized": float(routing_mode == "prompt_normalized"),
            "igsd/m2_turn_routing_is_top1_positive": float(routing_mode == "top1_positive"),
            "igsd/m2_distill_target_is_teacher_action": float(distill_target == "teacher_action"),
            "igsd/m2_distill_target_is_student_on_policy": float(distill_target == "student_on_policy"),
            "igsd/m2_routed_turn_count": 0.0,
            "igsd/m2_routed_prompt_count": 0.0,
            "igsd/m2_routed_turn_frac": 0.0,
            "igsd/row_verification_is_paired": float(not bypass_row_verification),
            "igsd/row_verification_is_bypass": float(bypass_row_verification),
            "igsd/row_teacher_generation_count": float(0 if bypass_row_verification else len(teacher_output)),
            "igsd/row_teacher_retrieval_query_count": 0.0,
            "igsd/row_ig_forward_count": 0.0,
        }
        alignment_valid_all = teacher_output.non_tensor_batch.get(
            "igsd_student_query_slot_alignment_valid"
        )
        if alignment_valid_all is not None:
            alignment_values = [bool(value) for value in alignment_valid_all.tolist()]
            metrics["igsd/student_query_slot_alignment_failed_count"] = float(
                sum(not value for value in alignment_values)
            )
            metrics["igsd/student_query_slot_alignment_valid_frac"] = float(
                np.mean(alignment_values) if alignment_values else 0.0
            )
        for turn_idx in sorted(set(source_turn_indices)):
            turn_rows = [idx for idx, value in enumerate(source_turn_indices) if value == turn_idx]
            metrics[f"igsd/teacher_turn_{turn_idx}_query_count"] = float(len(turn_rows))
            metrics[f"igsd/teacher_turn_{turn_idx}_valid_query_frac"] = float(
                np.mean([bool(teacher_queries[idx]) for idx in turn_rows])
            )
        if not valid_indices:
            return None, metrics

        teacher_query_documents_by_valid: list[list[list[str]]] = [[] for _ in valid_indices]
        if bypass_row_verification:
            valid_queries = ["" for _ in valid_indices]
            retrieved = [[] for _ in valid_indices]
        else:
            retrieval_url = str(self.config.algorithm.get("igsd_retrieval_url", ""))
            if not retrieval_url:
                if bool(self.config.algorithm.get("igsd_fail_on_missing_provider", False)):
                    raise ValueError("algorithm.igsd_retrieval_url is required for IGSD M2")
                metrics["igsd/teacher_missing_retrieval_url"] = 1.0
                return None, metrics
            valid_query_lists = [teacher_query_lists[idx] for idx in valid_indices]
            flat_valid_queries = [query for queries in valid_query_lists for query in queries]
            metrics["igsd/row_teacher_retrieval_query_count"] = float(len(flat_valid_queries))
            retrieved_flat = retrieve_sync(
                flat_valid_queries,
                retrieval_url,
                topk=int(self.config.algorithm.get("igsd_retrieval_topk", 3) or 3),
            )
            retrieved: list[list[str]] = []
            teacher_query_documents_by_valid = []
            cursor = 0
            for queries in valid_query_lists:
                query_docs = retrieved_flat[cursor : cursor + len(queries)]
                cursor += len(queries)
                teacher_query_documents_by_valid.append([list(docs) for docs in query_docs])
                retrieved.append([doc for docs in query_docs for doc in docs])
            retrieved_doc_counts = [len(docs) for docs in retrieved]
            nonempty_retrieval_count = sum(count > 0 for count in retrieved_doc_counts)
            metrics.update(
                {
                    "igsd/teacher_retrieval_query_count": float(len(flat_valid_queries)),
                    "igsd/teacher_retrieval_nonempty_count": float(nonempty_retrieval_count),
                    "igsd/teacher_retrieval_empty_count": float(
                        len(valid_query_lists) - nonempty_retrieval_count
                    ),
                    "igsd/teacher_retrieved_doc_count_mean": float(
                        np.mean(retrieved_doc_counts) if retrieved_doc_counts else 0.0
                    ),
                }
            )
            valid_source_turn_indices = [source_turn_indices[idx] for idx in valid_indices]
            for turn_idx in sorted(set(valid_source_turn_indices)):
                counts = [
                    count
                    for count, value in zip(retrieved_doc_counts, valid_source_turn_indices, strict=True)
                    if value == turn_idx
                ]
                metrics[f"igsd/teacher_turn_{turn_idx}_retrieval_nonempty_frac"] = float(
                    np.mean([count > 0 for count in counts])
                )
        valid_queries = [teacher_queries[idx] for idx in valid_indices]
        examples: list[dict[str, Any]] = []
        max_tool_response_tokens = int(
            self.config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        )
        tool_response_truncate_side = str(
            self.config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        )
        token_branch_records_all = teacher_output.non_tensor_batch.get("igsd_token_branch_records")
        token_budget_plans_all = teacher_output.non_tensor_batch.get("igsd_token_budget_plans")
        student_queries_all = teacher_output.non_tensor_batch.get("igsd_student_queries")
        student_slot_masks_all = teacher_output.non_tensor_batch.get("igsd_student_action_query_slot_mask")
        student_query_documents_all = teacher_output.non_tensor_batch.get("igsd_student_query_documents")
        bypass_teacher_prompt_ids = teacher_output.non_tensor_batch.get("igsd_teacher_prompt_ids")
        for output_idx, teacher_query, teacher_docs, teacher_query_documents in zip(
            valid_indices,
            valid_queries,
            retrieved,
            teacher_query_documents_by_valid,
            strict=True,
        ):
            source_sample_idx = int(teacher_output.non_tensor_batch["igsd_source_sample_idx"][output_idx])
            source_turn_idx = int(teacher_output.non_tensor_batch["igsd_source_turn_idx"][output_idx])
            entropy_record = (turn_entropy_records or {}).get((source_sample_idx, source_turn_idx), {})
            student_query = str(
                teacher_output.non_tensor_batch["igsd_student_query"][output_idx]
            )
            student_queries = (
                [str(query).strip() for query in list(student_queries_all[output_idx]) if str(query).strip()]
                if student_queries_all is not None
                else ([student_query] if student_query.strip() else [])
            )
            examples.append(
                {
                    "source_sample_idx": source_sample_idx,
                    "source_turn_idx": source_turn_idx,
                    "source_turn_count": int(
                        teacher_output.non_tensor_batch["igsd_source_turn_count"][output_idx]
                    ),
                    "prefix_ids": list(teacher_output.non_tensor_batch["igsd_prefix_ids"][output_idx]),
                    "teacher_prompt_ids": list(bypass_teacher_prompt_ids[output_idx])
                    if bypass_teacher_prompt_ids is not None
                    else clean_prompt_ids(
                        teacher_output,
                        output_idx,
                        int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0),
                    ),
                    "student_query": student_query,
                    "student_queries": student_queries,
                    "student_action_ids": list(
                        teacher_output.non_tensor_batch["igsd_student_action_ids"][output_idx]
                    ),
                    "student_action_query_mask": list(
                        teacher_output.non_tensor_batch["igsd_student_action_query_mask"][output_idx]
                    ),
                    "student_action_query_slot_mask": list(student_slot_masks_all[output_idx])
                    if student_slot_masks_all is not None
                    else [-1]
                    * len(teacher_output.non_tensor_batch["igsd_student_action_ids"][output_idx]),
                    "student_documents": list(
                        teacher_output.non_tensor_batch["igsd_student_documents"][output_idx]
                    ),
                    "student_query_documents": [
                        [str(doc) for doc in documents]
                        for documents in (
                            list(student_query_documents_all[output_idx])
                            if student_query_documents_all is not None
                            else []
                        )
                    ],
                    "student_tool_response_ids": list(
                        teacher_output.non_tensor_batch["igsd_student_tool_response_ids"][output_idx]
                    ),
                    "tool_response_template_text": str(
                        teacher_output.non_tensor_batch["igsd_tool_response_template_text"][output_idx]
                    ),
                    "tool_response_result_separator": result_separator,
                    "max_tool_response_tokens": max_tool_response_tokens,
                    "tool_response_truncate_side": tool_response_truncate_side,
                    "teacher_query": teacher_query or student_query,
                    "teacher_queries": list(teacher_query_lists[output_idx]) or student_queries,
                    "teacher_query_documents": teacher_query_documents,
                    "teacher_documents": list(teacher_docs),
                    "igsd_token_branch_records": deepcopy(
                        token_branch_records_all[output_idx] if token_branch_records_all is not None else []
                    ),
                    "igsd_token_budget_plan": deepcopy(
                        token_budget_plans_all[output_idx] if token_budget_plans_all is not None else []
                    ),
                    "answer_aliases": list(teacher_output.non_tensor_batch["igsd_answer_aliases"][output_idx]),
                    "success_queries": list(teacher_output.non_tensor_batch["igsd_success_queries"][output_idx]),
                    "student_action_entropy": entropy_record.get("action_entropy"),
                    "student_action_token_count": entropy_record.get("action_token_count"),
                    "student_query_entropy": entropy_record.get("query_entropy"),
                    "student_query_token_count": entropy_record.get("query_token_count"),
                    "student_non_query_entropy": entropy_record.get("non_query_entropy"),
                    "student_non_query_token_count": entropy_record.get("non_query_token_count"),
                }
            )
        metrics["igsd/teacher_retrieval_nonempty_frac"] = float(
            np.mean([bool(ex["teacher_documents"]) for ex in examples]) if examples else 0.0
        )
        metrics["igsd/m2_student_action_query_alignment_frac"] = float(
            np.mean([any(ex["student_action_query_mask"]) for ex in examples]) if examples else 0.0
        )
        if bypass_row_verification:
            paired = [
                {
                    **ex,
                    "ig_student": 0.0,
                    "ig_teacher": 0.0,
                    "ig_delta": 0.0,
                    "ig_gate_signal": 0.0,
                    "ig_gate": 1.0,
                    "ig_turn_weight": 1.0,
                    "ig_row_verification_valid": False,
                }
                for ex in examples
            ]
            metrics.update(
                {
                    "igsd/m2_paired_ig_bypassed": 1.0,
                    "igsd/m2_valid_pair_count": 0.0,
                    "igsd/m2_pair_coverage": 0.0,
                    "igsd/m2_pseudo_sequence_count": 0.0,
                }
            )
        else:
            paired, paired_metrics = compute_paired_query_ig(
                examples=examples,
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=self.config.algorithm,
                global_step=self.global_steps,
                log_prob_micro_batch_size_per_gpu=(
                    self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                ),
            )
            metrics.update(paired_metrics)
            metrics["igsd/row_ig_forward_count"] = float(len(paired))
        token_weight_mode = str(
            self.config.algorithm.get("igsd_token_weight_mode", "uniform")
        ).lower()
        gate_source = str(self.config.algorithm.get("igsd_gate_source", "paired_ig")).lower()
        preserve_zero_token_gate_rows = bool(
            self.config.algorithm.get("igsd_preserve_zero_token_gate_rows", False)
        )
        if token_weight_mode in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }:
            branch_queries: list[str] = []
            branch_locations: list[tuple[int, int, int]] = []
            student_queries_to_retrieve: list[str] = []
            student_query_locations: list[tuple[int, int]] = []
            branch_retrieval_url = str(self.config.algorithm.get("igsd_retrieval_url", ""))
            for example_idx, ex in enumerate(paired):
                records = ex.get("igsd_token_branch_records", [])
                for record_idx, record in enumerate(records):
                    if record.get("branch_kind") not in {
                        "teacher_continuation",
                        "budgeted_teacher_top1",
                        "budgeted_teacher_top2",
                        "sampled_pair_teacher",
                        "sampled_pair_student",
                    }:
                        continue
                    queries = [
                        str(query).strip()
                        for query in record.get("branch_queries", [])
                        if str(query).strip()
                    ]
                    if not queries:
                        legacy_query = str(record.get("branch_query", "")).strip()
                        queries = [legacy_query] if legacy_query else []
                    if not record.get("branch_valid") or not queries:
                        continue
                    for query_idx, query in enumerate(queries):
                        branch_queries.append(query)
                        branch_locations.append((example_idx, record_idx, query_idx))
                    record["branch_queries"] = queries
                    record["branch_query"] = " ".join(queries)
                for record in records:
                    if record.get("branch_kind") == "student_terminal":
                        student_docs = list(ex.get("student_documents", []))
                        student_query_lists = ex.get("student_queries", [])
                        student_query_documents = [
                            list(documents)
                            for documents in ex.get("student_query_documents", [])
                        ]
                        # Summary-enabled ASearcher turns often have no raw
                        # documents in the decoded trajectory.  Re-query the
                        # complete student query_list so the terminal branch
                        # remains comparable to candidate branches.
                        if not student_docs and student_query_lists:
                            for query_idx, query in enumerate(student_query_lists):
                                if str(query).strip():
                                    student_queries_to_retrieve.append(str(query).strip())
                                    student_query_locations.append((example_idx, query_idx))
                        record["branch_queries"] = list(student_query_lists)
                        record["branch_query"] = " ".join(student_query_lists)
                        record["branch_query_documents"] = student_query_documents
                        record["branch_documents"] = [
                            doc
                            for query_docs in student_query_documents
                            for doc in query_docs
                        ] or student_docs
            if branch_queries and branch_retrieval_url:
                branch_retrieved = retrieve_sync(
                    branch_queries,
                    branch_retrieval_url,
                    topk=int(self.config.algorithm.get("igsd_retrieval_topk", 3) or 3),
                )
                if len(branch_retrieved) != len(branch_locations):
                    raise RuntimeError(
                        "IGSD branch retrieval output count must match query locations, "
                        f"got {len(branch_retrieved)} results for {len(branch_locations)} queries"
                    )
                grouped_branch_documents: dict[tuple[int, int], list[list[str]]] = {}
                for (example_idx, record_idx, query_idx), documents in zip(
                    branch_locations, branch_retrieved, strict=True
                ):
                    key = (example_idx, record_idx)
                    record = paired[example_idx]["igsd_token_branch_records"][record_idx]
                    expected_query_count = len(record.get("branch_queries", []))
                    if key not in grouped_branch_documents:
                        grouped_branch_documents[key] = [
                            [] for _ in range(expected_query_count)
                        ]
                    if not 0 <= query_idx < expected_query_count:
                        raise RuntimeError(
                            "IGSD branch retrieval query index is outside its query_list, "
                            f"got {query_idx=} for {expected_query_count=}"
                        )
                    grouped_branch_documents[key][query_idx] = list(documents)
                for (example_idx, record_idx), query_documents in grouped_branch_documents.items():
                    record = paired[example_idx]["igsd_token_branch_records"][record_idx]
                    record["branch_query_documents"] = query_documents
                    record["branch_documents"] = [doc for docs in query_documents for doc in docs]
                metrics["igsd/token_intervention_retrieval_query_count"] = float(len(branch_queries))
                metrics["igsd/token_intervention_retrieval_nonempty_frac"] = float(
                    np.mean([bool(documents) for documents in branch_retrieved])
                )
            elif branch_queries:
                metrics["igsd/token_intervention_missing_retrieval_url"] = 1.0
                metrics["igsd/token_intervention_retrieval_query_count"] = float(len(branch_queries))
                metrics["igsd/token_intervention_retrieval_nonempty_frac"] = 0.0
            else:
                metrics["igsd/token_intervention_retrieval_query_count"] = 0.0
                metrics["igsd/token_intervention_retrieval_nonempty_frac"] = 0.0
            if student_queries_to_retrieve and branch_retrieval_url:
                student_retrieved = retrieve_sync(
                    student_queries_to_retrieve,
                    branch_retrieval_url,
                    topk=int(self.config.algorithm.get("igsd_retrieval_topk", 3) or 3),
                )
                if len(student_retrieved) != len(student_query_locations):
                    raise RuntimeError(
                        "IGSD student retrieval output count must match query locations, "
                        f"got {len(student_retrieved)} results for "
                        f"{len(student_query_locations)} queries"
                    )
                grouped_student_documents: dict[int, list[list[str]]] = {}
                for (example_idx, query_idx), documents in zip(
                    student_query_locations, student_retrieved, strict=True
                ):
                    expected_query_count = len(paired[example_idx].get("student_queries", []))
                    if example_idx not in grouped_student_documents:
                        grouped_student_documents[example_idx] = [
                            [] for _ in range(expected_query_count)
                        ]
                    if not 0 <= query_idx < expected_query_count:
                        raise RuntimeError(
                            "IGSD student retrieval query index is outside its query_list, "
                            f"got {query_idx=} for {expected_query_count=}"
                        )
                    grouped_student_documents[example_idx][query_idx] = list(documents)
                for example_idx, query_documents in grouped_student_documents.items():
                    for record in paired[example_idx].get("igsd_token_branch_records", []):
                        if record.get("branch_kind") == "student_terminal":
                            record["branch_query_documents"] = query_documents
                            record["branch_documents"] = [
                                doc for docs in query_documents for doc in docs
                            ]
                metrics["igsd/student_terminal_retrieval_query_count"] = float(
                    len(student_queries_to_retrieve)
                )
                metrics["igsd/student_terminal_retrieval_nonempty_frac"] = float(
                    np.mean([bool(documents) for documents in student_retrieved])
                    if student_retrieved else 0.0
                )
            elif student_queries_to_retrieve:
                metrics["igsd/student_terminal_missing_retrieval_url"] = 1.0
                metrics["igsd/student_terminal_retrieval_query_count"] = float(
                    len(student_queries_to_retrieve)
                )
                metrics["igsd/student_terminal_retrieval_nonempty_frac"] = 0.0
            else:
                metrics["igsd/student_terminal_retrieval_query_count"] = 0.0
                metrics["igsd/student_terminal_retrieval_nonempty_frac"] = float(
                    np.mean(
                        [
                            bool(record.get("branch_documents"))
                            for ex in paired
                            for record in ex.get("igsd_token_branch_records", [])
                            if record.get("branch_kind") == "student_terminal"
                        ]
                    )
                    if paired else 0.0
                )
            paired, token_ig_metrics = compute_token_intervention_ig(
                examples=paired,
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=self.config.algorithm,
                global_step=self.global_steps,
                log_prob_micro_batch_size_per_gpu=(
                    self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                ),
            )
            metrics.update(token_ig_metrics)
        logged_count = self._dump_igsd_m2_examples(examples=examples, paired=paired)
        metrics["igsd/m2_logged_example_count"] = float(logged_count)
        metrics["igsd/m2_logged_gate_is_final"] = float(
            gate_source == "paired_ig" and row_verification_mode != "bypass"
        )
        active_paired = [
            ex for ex in paired if float(ex.get("ig_turn_weight", ex.get("ig_gate", 0.0))) > 0.0
        ]
        if token_weight_mode == "sampled_action_pair_ig":
            sampled_pair_gate_source = str(
                self.config.algorithm.get("igsd_sampled_pair_gate_source", "environment_ig")
            ).lower()
            sampled_positive_rows = sum(
                any(float(value) > 0.0 for value in ex.get("igsd_token_gate", []))
                for ex in paired
            )
            # Row bypass uses a constant carrier weight of one, so ``active_paired``
            # contains every source row even when no token pair passes the configured
            # gate. Report actual positive-token rows under the generic active name;
            # the explicit carrier metric below retains the denominator population.
            active_distill_pair_count = sampled_positive_rows
            metrics["igsd/sampled_pair_carrier_row_count"] = float(len(paired))
            metrics["igsd/sampled_pair_active_row_count"] = float(sampled_positive_rows)
            metrics["igsd/sampled_pair_active_row_frac"] = float(
                sampled_positive_rows / max(len(paired), 1)
            )
            metrics["igsd/sampled_pair_verified_positive_row_count"] = float(
                sampled_positive_rows if sampled_pair_gate_source == "environment_ig" else 0
            )
            metrics["igsd/sampled_pair_verified_positive_row_frac"] = float(
                sampled_positive_rows / max(len(paired), 1)
                if sampled_pair_gate_source == "environment_ig"
                else 0.0
            )
        else:
            active_distill_pair_count = len(active_paired)
        metrics["igsd/m2_active_distill_pair_count"] = float(active_distill_pair_count)
        metrics["igsd/m2_active_distill_pair_frac"] = float(
            active_distill_pair_count / max(len(paired), 1)
        )
        # Legacy independent routing keeps its original auxiliary-batch shape,
        # including zero-gate rows. New routing modes can safely drop rejected
        # turns because their selection semantics explicitly exclude those rows.
        routing_mode = str(self.config.algorithm.get("igsd_turn_routing_mode", "independent")).lower()
        actor_paired = paired if routing_mode == "independent" else active_paired
        if token_weight_mode in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }:
            before_token_valid_filter = len(actor_paired)
            all_invalid_rows = 0
            all_zero_gate_rows = 0
            zero_turn_weight_rows = 0
            filtered_actor_paired = []
            for ex in actor_paired:
                valid_values = [bool(value) for value in ex.get("igsd_token_gate_valid", [])]
                gate_values = [float(value) for value in ex.get("igsd_token_gate", [])]
                if len(valid_values) != len(gate_values):
                    raise ValueError(
                        "Prefix-intervention token gate fields must have identical lengths before filtering"
                    )
                has_valid = any(valid_values)
                has_positive = any(
                    is_valid and gate > 0.0 for is_valid, gate in zip(valid_values, gate_values, strict=True)
                )
                if not has_valid:
                    all_invalid_rows += 1
                elif not has_positive:
                    all_zero_gate_rows += 1
                has_positive_turn_weight = (
                    float(ex.get("ig_turn_weight", ex.get("ig_gate", 0.0))) > 0.0
                )
                if has_positive and not has_positive_turn_weight:
                    zero_turn_weight_rows += 1
                if (has_positive or preserve_zero_token_gate_rows) and has_positive_turn_weight:
                    filtered_actor_paired.append(ex)
            actor_paired = filtered_actor_paired
            metrics["igsd/token_intervention_all_invalid_filtered_count"] = float(
                all_invalid_rows
            )
            metrics["igsd/token_intervention_all_zero_gate_filtered_count"] = float(
                all_zero_gate_rows
            )
            metrics["igsd/token_intervention_zero_turn_weight_filtered_count"] = float(
                zero_turn_weight_rows
            )
            metrics["igsd/token_intervention_effective_filter_count"] = float(
                before_token_valid_filter - len(actor_paired)
            )
        on_policy_aligned_count = 0
        on_policy_length_filtered_count = 0
        if distill_target == "student_on_policy":
            actor_paired = [ex for ex in actor_paired if any(ex["student_action_query_mask"])]
            on_policy_aligned_count = len(actor_paired)
            max_seq_len = int(
                self.config.algorithm.get(
                    "igsd_max_pseudo_seq_len",
                    self.config.actor_rollout_ref.rollout.max_model_len,
                )
                or self.config.actor_rollout_ref.rollout.max_model_len
            )
            if max_seq_len > 0:
                before_length_filter = len(actor_paired)
                actor_paired = [
                    ex
                    for ex in actor_paired
                    if max(len(ex["prefix_ids"]), len(ex["teacher_prompt_ids"]))
                    + len(ex["student_action_ids"])
                    <= max_seq_len
                ]
                on_policy_length_filtered_count = before_length_filter - len(actor_paired)
        metrics["igsd/m2_actor_aux_pair_count"] = float(len(actor_paired))
        metrics["igsd/m2_on_policy_aligned_pair_count"] = float(on_policy_aligned_count)
        metrics["igsd/m2_on_policy_length_filtered_count"] = float(on_policy_length_filtered_count)
        lambda_value = self.config.algorithm.get("igsd_lambda", 0.1)
        lambda_coef = 0.1 if lambda_value is None else float(lambda_value)
        if lambda_coef <= 0.0:
            metrics.update(
                {
                    "igsd/m2_bc_sample_count": 0.0,
                    "igsd/m2_distill_span_is_query_only": float(distill_span == "query_only"),
                    "igsd/m2_distill_span_token_frac": 0.0,
                    "igsd/m2_metrics_only": 1.0,
                }
            )
            return None, metrics
        metrics["igsd/m2_metrics_only"] = 0.0
        teacher_log_probs = None
        teacher_topk_ids = None
        teacher_topk_log_probs = None
        teacher_topk_mask = None
        if distill_mode == "candidate_pair_jsd" and actor_paired:
            teacher_topk_ids, teacher_topk_log_probs, teacher_topk_mask = (
                build_sampled_action_pair_targets(
                    actor_paired,
                    utility_margin=float(
                        self.config.algorithm.get("igsd_token_gate_margin", 0.0)
                    ),
                )
            )
            metrics.update(
                {
                    "igsd/candidate_pair_target_position_count": float(teacher_topk_mask.numel()),
                    "igsd/candidate_pair_target_active_position_count": float(
                        teacher_topk_mask.sum().item()
                    ),
                    "igsd/candidate_pair_target_active_frac": float(
                        teacher_topk_mask.float().mean().item()
                        if teacher_topk_mask.numel()
                        else 0.0
                    ),
                    "igsd/candidate_pair_support_width": 2.0,
                }
            )
        elif distill_mode in {"topk_reverse_kl", "topk_jsd"} and actor_paired:
            student_topk_batch = build_action_logprob_batch(
                actor_paired,
                self.tokenizer,
                prompt_key="prefix_ids",
                distill_target=distill_target,
            )
            if student_topk_batch is None:
                raise ValueError("IGSD top-k distillation requires student top-k samples")
            kept = student_topk_batch.non_tensor_batch.get("igsd_kept_example_idx")
            if kept is not None and len(kept) != len(actor_paired):
                kept_indices = [int(idx) for idx in kept.tolist()]
                actor_paired = [actor_paired[idx] for idx in kept_indices]
            sampled_pair_topk_mask = None
            if token_weight_mode == "sampled_action_pair_ig":
                # The environment scorer has already populated the final
                # positive Delta-IG state on each example. Restrict both
                # target forwards to those positions so inactive query tokens
                # never require full-vocabulary top-k materialization.
                sampled_pair_topk_mask = build_sampled_action_positive_mask(
                    actor_paired,
                    response_width=int(student_topk_batch.batch["responses"].shape[1]),
                    utility_margin=float(
                        self.config.algorithm.get("igsd_token_gate_margin", 0.0)
                    ),
                )
                if sampled_pair_topk_mask.shape != student_topk_batch.batch["responses"].shape:
                    raise ValueError(
                        "Sampled-action top-k positive mask must align with the student target response: "
                        f"{sampled_pair_topk_mask.shape=} and "
                        f"{student_topk_batch.batch['responses'].shape=}"
                    )
            student_topk_start = time.perf_counter()
            student_topk_ids, _, student_topk_mfu = self._compute_igsd_action_topk(
                student_topk_batch,
                topk=topk,
                response_selection_mask=(
                    sampled_pair_topk_mask
                    if sampled_pair_topk_mask is not None
                    else student_topk_batch.batch.get("igsd_action_query_mask")
                    if distill_target == "student_on_policy" and distill_span == "query_only"
                    else None
                ),
            )
            student_topk_seconds = time.perf_counter() - student_topk_start

            teacher_topk_batch = build_action_logprob_batch(
                actor_paired,
                self.tokenizer,
                prompt_key="teacher_prompt_ids",
                distill_target=distill_target,
            )
            if teacher_topk_batch is None:
                raise ValueError("IGSD top-k distillation requires teacher top-k samples")
            kept = teacher_topk_batch.non_tensor_batch.get("igsd_kept_example_idx")
            if kept is not None and len(kept) != len(actor_paired):
                kept_indices = [int(idx) for idx in kept.tolist()]
                actor_paired = [actor_paired[idx] for idx in kept_indices]
                student_topk_ids = student_topk_ids[kept_indices]
                if sampled_pair_topk_mask is not None:
                    sampled_pair_topk_mask = sampled_pair_topk_mask[kept_indices]
            if sampled_pair_topk_mask is not None:
                if sampled_pair_topk_mask.shape[1] != teacher_topk_batch.batch["responses"].shape[1]:
                    raise ValueError(
                        "Sampled-action top-k positive mask must align with the teacher target response: "
                        f"{sampled_pair_topk_mask.shape=} and "
                        f"{teacher_topk_batch.batch['responses'].shape=}"
                    )
            teacher_topk_start = time.perf_counter()
            teacher_topk_ids, teacher_topk_log_probs, teacher_topk_mfu = self._compute_igsd_action_topk(
                teacher_topk_batch,
                topk=topk,
                candidate_ids=student_topk_ids,
                response_selection_mask=(
                    sampled_pair_topk_mask
                    if sampled_pair_topk_mask is not None
                    else teacher_topk_batch.batch.get("igsd_action_query_mask")
                    if distill_target == "student_on_policy" and distill_span == "query_only"
                    else None
                ),
            )
            teacher_topk_seconds = time.perf_counter() - teacher_topk_start
            if token_weight_mode == "budgeted_local_candidate_ig":
                if teacher_topk_log_probs is None:
                    raise RuntimeError("Budgeted local candidates require teacher top-k candidate log-probs")
                teacher_topk_log_probs, tilt_metrics = apply_budgeted_local_candidate_target_tilts(
                    teacher_topk_ids,
                    teacher_topk_log_probs,
                    actor_paired,
                    eta=float(self.config.algorithm.get("igsd_budgeted_target_eta", 1.0)),
                )
                metrics.update(tilt_metrics)
            elif token_weight_mode == "sampled_action_pair_ig":
                if distill_mode not in {"topk_jsd", "topk_reverse_kl"}:
                    raise AssertionError(
                        "sampled_action_pair_ig reaches the generic top-k target path only for "
                        "topk_jsd or topk_reverse_kl"
                    )
                if sampled_pair_topk_mask is None:
                    raise AssertionError("sampled-action top-k mode did not build its positive target mask")
                # Reuse the same mask that restricted both target forwards.
                # Inactive query positions retain zero targets and never
                # materialize actor logits in the loss path.
                teacher_topk_mask = sampled_pair_topk_mask
                metrics.update(
                    {
                        "igsd/sampled_pair_topk_target_requested_position_count": float(
                            teacher_topk_mask.sum().item()
                        ),
                        "igsd/sampled_pair_topk_target_active_position_count": float(
                            teacher_topk_mask.sum().item()
                        ),
                        "igsd/sampled_pair_topk_target_active_frac": float(
                            teacher_topk_mask.float().mean().item()
                            if teacher_topk_mask.numel()
                            else 0.0
                        ),
                    }
                )
            metrics.update(
                {
                    "igsd/topk": float(topk),
                    "igsd/topk_support_width": float(2 * topk),
                    "igsd/topk_target_materialization_chunk_size": float(
                        materialization_chunk_size
                    ),
                    "igsd/student_topk_mfu": student_topk_mfu,
                    "igsd/teacher_topk_mfu": teacher_topk_mfu,
                    "timing_s/igsd_student_topk_targets": student_topk_seconds,
                    "timing_s/igsd_teacher_topk_targets": teacher_topk_seconds,
                    "igsd/topk_target_token_count": float(
                        teacher_topk_batch.batch["response_mask"].sum().item()
                    ),
                    "igsd/topk_distilled_query_token_count": float(
                        sum(sum(ex["student_action_query_mask"]) for ex in actor_paired)
                        if distill_target == "student_on_policy"
                        else 0
                    ),
                    "igsd/topk_positive_verified_query_token_count": float(
                        teacher_topk_mask.sum().item()
                        if teacher_topk_mask is not None and token_weight_mode == "sampled_action_pair_ig"
                        else 0.0
                    ),
                }
            )
        elif distill_mode not in {"bc", "nll", "candidate_pair_jsd"} and actor_paired:
            teacher_logprob_batch = build_action_logprob_batch(
                actor_paired,
                self.tokenizer,
                prompt_key="teacher_prompt_ids",
                distill_target=distill_target,
            )
            if teacher_logprob_batch is None:
                raise ValueError(f"IGSD distill mode {distill_mode!r} requires teacher logprob samples")
            teacher_logprob_proto, teacher_logprob_mfu = self._compute_old_log_prob_padded(teacher_logprob_batch)
            teacher_log_probs = teacher_logprob_proto.batch["old_log_probs"].detach().cpu()
            kept = teacher_logprob_batch.non_tensor_batch.get("igsd_kept_example_idx")
            if kept is not None and len(kept) != len(actor_paired):
                kept_indices = [int(idx) for idx in kept.tolist()]
                actor_paired = [actor_paired[idx] for idx in kept_indices]
            metrics.update(
                {
                    "igsd/teacher_soft_log_prob_mfu": float(teacher_logprob_mfu),
                    "igsd/teacher_soft_log_prob_count": float(teacher_log_probs.numel()),
                    "igsd/teacher_soft_log_prob_mean": float(
                        teacher_log_probs[teacher_log_probs != 0].mean().item()
                    )
                    if bool((teacher_log_probs != 0).any())
                    else 0.0,
                }
            )
        bc_batch = build_student_bc_batch(
            actor_paired,
            self.tokenizer,
            lambda_coef=lambda_coef,
            distill_span=distill_span,
            distill_target=distill_target,
            token_weight_mode=token_weight_mode,
            preserve_zero_token_gate_rows=preserve_zero_token_gate_rows,
            distill_mode=distill_mode,
            teacher_log_probs=teacher_log_probs,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
            teacher_topk_mask=teacher_topk_mask,
        )
        metrics["igsd/m2_bc_sample_count"] = float(0 if bc_batch is None else len(bc_batch))
        metrics["igsd/m2_distill_span_is_query_only"] = float(distill_span == "query_only")
        if bc_batch is not None:
            valid_response_tokens = bc_batch.batch["response_mask"].bool()
            distill_span_tokens = bc_batch.batch["igsd_query_distill_mask"].float()
            metrics["igsd/m2_distill_span_token_frac"] = (
                float(distill_span_tokens[valid_response_tokens].mean().detach().cpu().item())
                if bool(valid_response_tokens.any())
                else 0.0
            )
            if token_weight_mode in {
                "prefix_intervention_ig",
                "budgeted_local_candidate_ig",
                "sampled_action_pair_ig",
            }:
                token_valid = (
                    bc_batch.batch["igsd_token_gate_valid"].bool()
                    & bc_batch.batch["igsd_query_token_mask"].bool()
                    & valid_response_tokens
                )
                weights = bc_batch.batch["igsd_distill_weights"].float()
                positive = token_valid & (weights > 0.0)
                metrics.update(
                    {
                        "igsd/token_intervention_projected_valid_token_count": float(
                            token_valid.sum().item()
                        ),
                        "igsd/token_intervention_effective_distill_token_count": float(
                            positive.sum().item()
                        ),
                        "igsd/token_intervention_effective_distill_weight_sum": float(
                            weights[token_valid].sum().detach().cpu().item()
                        )
                        if bool(token_valid.any())
                        else 0.0,
                        "igsd/token_intervention_effective_distill_weight_mean": float(
                            weights[token_valid].mean().detach().cpu().item()
                        )
                        if bool(token_valid.any())
                        else 0.0,
                        "igsd/token_intervention_effective_distill_weight_max": float(
                            weights[token_valid].max().detach().cpu().item()
                        )
                        if bool(token_valid.any())
                        else 0.0,
                        "igsd/token_intervention_effective_distill_row_count": float(
                            positive.any(dim=1).sum().item()
                        ),
                    }
                )
        else:
            metrics["igsd/m2_distill_span_token_frac"] = 0.0
        return bc_batch, metrics

    def _dump_igsd_m2_examples(self, examples: list[dict[str, Any]], paired: list[dict[str, Any]]) -> int:
        max_examples = int(self.config.algorithm.get("igsd_log_examples", 0) or 0)
        if max_examples <= 0 or not examples:
            return 0

        def example_key(ex: dict[str, Any]) -> tuple[Any, Any, str, str]:
            return (
                ex.get("source_sample_idx"),
                ex.get("source_turn_idx"),
                str(ex.get("student_query", "")),
                str(ex.get("teacher_query", "")),
            )

        paired_by_key = {example_key(ex): ex for ex in paired}
        gate_source = str(self.config.algorithm.get("igsd_gate_source", "paired_ig")).lower()
        row_verification_mode = str(
            self.config.algorithm.get("igsd_row_verification_mode", "paired")
        ).lower()

        def preview_docs(docs: list[str], limit: int = 3, chars: int = 260) -> list[str]:
            previews = []
            for doc in docs[:limit]:
                text = " ".join(str(doc).split())
                previews.append(text[:chars])
            return previews

        def summarize_branch(record: dict[str, Any]) -> dict[str, Any]:
            return {
                "branch_index": record.get("branch_index"),
                "branch_kind": record.get("branch_kind"),
                "action_query_position": record.get("action_query_position"),
                "prompt_length": record.get("prompt_length"),
                "generated_token_count": record.get("generated_token_count"),
                "max_new_tokens": record.get("max_new_tokens"),
                "hit_generation_limit": record.get("hit_generation_limit", False),
                "stop_reason": record.get("stop_reason"),
                "branch_query": record.get("branch_query", ""),
                "branch_queries": record.get("branch_queries", []),
                "branch_valid": record.get("branch_valid", False),
                "branch_error": record.get("branch_error"),
                "pair_side": record.get("pair_side"),
                "candidate_token_id": record.get("candidate_token_id"),
                "sampled_token_id": record.get("sampled_token_id"),
                "reference_token_id": record.get("reference_token_id"),
                "teacher_top1_id": record.get("teacher_top1_id"),
                "student_top1_id": record.get("student_top1_id"),
                "pair_log_odds_shift": record.get("pair_log_odds_shift"),
                "branch_doc_count": len(record.get("branch_documents", [])),
                "branch_query_doc_counts": [
                    len(docs) for docs in record.get("branch_query_documents", [])
                ],
                "branch_ig": record.get("branch_ig"),
                "branch_ig_valid": record.get("branch_ig_valid", False),
                "branch_ig_alias_mode": record.get("branch_ig_alias_mode"),
            }

        rows = []
        for ex in examples[:max_examples]:
            paired_ex = paired_by_key.get(example_key(ex), {})
            token_debug_ex = paired_ex if paired_ex else ex
            row_ig_computed = bool(paired_ex) and row_verification_mode != "bypass"
            source_turn_idx = int(ex.get("source_turn_idx", -1))
            source_turn_count = int(ex.get("source_turn_count", 1))
            rows.append(
                {
                    "global_step": int(self.global_steps),
                    "source_sample_idx": int(ex.get("source_sample_idx", -1)),
                    "source_turn_idx": source_turn_idx,
                    "source_turn_count": source_turn_count,
                    "source_relative_turn": paired_ex.get(
                        "source_relative_turn",
                        source_turn_idx / max(source_turn_count - 1, 1),
                    ),
                    "source_turn_bucket": paired_ex.get("source_turn_bucket"),
                    "student_query": ex.get("student_query", ""),
                    "teacher_query": ex.get("teacher_query", ""),
                    "teacher_query_changed": str(ex.get("teacher_query", "")).strip()
                    != str(ex.get("student_query", "")).strip(),
                    "student_action_entropy": ex.get("student_action_entropy"),
                    "student_action_token_count": ex.get("student_action_token_count"),
                    "student_query_entropy": ex.get("student_query_entropy"),
                    "student_query_token_count": ex.get("student_query_token_count"),
                    "student_non_query_entropy": ex.get("student_non_query_entropy"),
                    "student_non_query_token_count": ex.get("student_non_query_token_count"),
                    "success_queries": ex.get("success_queries", []),
                    "answer_aliases": ex.get("answer_aliases", []),
                    "student_doc_count": len(ex.get("student_documents", [])),
                    "teacher_doc_count": len(ex.get("teacher_documents", [])),
                    "student_doc_previews": preview_docs(ex.get("student_documents", [])),
                    "teacher_doc_previews": preview_docs(ex.get("teacher_documents", [])),
                    "ig_computed": row_ig_computed,
                    "row_verification_valid": row_ig_computed,
                    "ig_student": paired_ex.get("ig_student") if row_ig_computed else None,
                    "ig_teacher": paired_ex.get("ig_teacher") if row_ig_computed else None,
                    "ig_delta": paired_ex.get("ig_delta") if row_ig_computed else None,
                    "ig_cf_count": paired_ex.get("ig_cf_count") if row_ig_computed else None,
                    "ig_cf_delta_std": paired_ex.get("ig_cf_delta_std") if row_ig_computed else None,
                    "ig_cf_delta_se": paired_ex.get("ig_cf_delta_se") if row_ig_computed else None,
                    "ig_cf_delta_positive_frac": (
                        paired_ex.get("ig_cf_delta_positive_frac") if row_ig_computed else None
                    ),
                    "ig_cf_delta_sign_agreement": (
                        paired_ex.get("ig_cf_delta_sign_agreement") if row_ig_computed else None
                    ),
                    "ig_lcb_delta": paired_ex.get("ig_lcb_delta") if row_ig_computed else None,
                    "ig_lcb_valid": paired_ex.get("ig_lcb_valid") if row_ig_computed else None,
                    "ig_gate_signal": paired_ex.get("ig_gate_signal") if row_ig_computed else None,
                    "is_max_delta_turn": paired_ex.get("is_max_delta_turn"),
                    "training_gate_source": gate_source,
                    "sampled_pair_gate_source": str(
                        self.config.algorithm.get("igsd_sampled_pair_gate_source", "environment_ig")
                    ).lower(),
                    "sampled_pair_gate_granularity": str(
                        self.config.algorithm.get("igsd_sampled_pair_gate_granularity", "token")
                    ).lower(),
                    "training_gate_available_in_dump": (
                        gate_source == "paired_ig"
                        and row_verification_mode != "bypass"
                        and bool(paired_ex)
                    ),
                    "paired_ig_gate": paired_ex.get("ig_gate") if row_ig_computed else None,
                    "paired_ig_turn_weight": (
                        paired_ex.get("ig_turn_weight") if row_ig_computed else None
                    ),
                    "paired_ig_turn_selected": (
                        paired_ex.get("ig_turn_selected") if row_ig_computed else None
                    ),
                    "ig_gate": (
                        paired_ex.get("ig_gate")
                        if gate_source == "paired_ig" and row_ig_computed
                        else None
                    ),
                    "ig_turn_weight": paired_ex.get("ig_turn_weight")
                    if gate_source == "paired_ig" and row_ig_computed
                    else None,
                    "ig_turn_selected": paired_ex.get("ig_turn_selected")
                    if gate_source == "paired_ig" and row_ig_computed
                    else None,
                    "token_intervention_branches": [
                        summarize_branch(record)
                        for record in token_debug_ex.get("igsd_token_branch_records", [])
                    ],
                    "token_intervention_ig_values": token_debug_ex.get("igsd_token_ig_values", []),
                    "token_intervention_branch_ig_valid": token_debug_ex.get(
                        "igsd_token_branch_ig_valid", []
                    ),
                    "token_intervention_gains": token_debug_ex.get("igsd_token_gains", []),
                    "token_intervention_gate_signals": token_debug_ex.get(
                        "igsd_token_gate_signals",
                        token_debug_ex.get("igsd_token_gains", []),
                    ),
                    "token_intervention_pair_left_ig": token_debug_ex.get(
                        "igsd_token_pair_left_ig", []
                    ),
                    "token_intervention_pair_right_ig": token_debug_ex.get(
                        "igsd_token_pair_right_ig", []
                    ),
                    "token_intervention_common_cf_count": token_debug_ex.get(
                        "igsd_token_common_cf_count", []
                    ),
                    "token_intervention_common_alias_count_min": token_debug_ex.get(
                        "igsd_token_common_alias_count_min", []
                    ),
                    "token_intervention_common_alias_count_mean": token_debug_ex.get(
                        "igsd_token_common_alias_count_mean", []
                    ),
                    "token_intervention_gate": token_debug_ex.get("igsd_token_gate", []),
                    "token_intervention_gate_valid": token_debug_ex.get(
                        "igsd_token_gate_valid", []
                    ),
                    # ``delta_valid`` records whether paired evidence utility
                    # is available; ``loss_active`` is the post-gate label.
                    "token_intervention_delta_valid": token_debug_ex.get(
                        "igsd_token_delta_valid", []
                    ),
                    "token_intervention_loss_active": [
                        bool(is_valid) and float(gate_value) > 0.0
                        for is_valid, gate_value in zip(
                            token_debug_ex.get("igsd_token_gate_valid", []),
                            token_debug_ex.get("igsd_token_gate", []),
                            strict=True,
                        )
                    ],
                    "sampled_pair_plans": token_debug_ex.get("igsd_token_budget_plan", []),
                }
            )

        dump_root = os.path.join(str(self.config.trainer.default_local_dir), "igsd_debug")
        os.makedirs(dump_root, exist_ok=True)
        filename = os.path.join(dump_root, f"{self.global_steps}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return len(rows)

    def _pad_prompt_width_for_concat(self, data: DataProto, target_prompt_len: int) -> DataProto:
        prompt_len = data.batch["prompts"].shape[1]
        pad_len = target_prompt_len - prompt_len
        if pad_len <= 0:
            return data

        pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        device = data.batch["prompts"].device
        prompt_pad = torch.full((len(data), pad_len), pad_id, dtype=data.batch["prompts"].dtype, device=device)
        data.batch["prompts"] = torch.cat([data.batch["prompts"], prompt_pad], dim=1)

        for key in ("input_ids", "attention_mask"):
            if key not in data.batch:
                continue
            tensor = data.batch[key]
            left = tensor[:, :prompt_len]
            right = tensor[:, prompt_len:]
            fill = pad_id if key == "input_ids" else 0
            pad = torch.full((len(data), pad_len), fill, dtype=tensor.dtype, device=tensor.device)
            data.batch[key] = torch.cat([left, pad, right], dim=1)

        if "position_ids" in data.batch:
            position_ids = data.batch["position_ids"]
            if position_ids.dim() == 2:
                left = position_ids[:, :prompt_len]
                right = position_ids[:, prompt_len:]
                pad = torch.zeros((len(data), pad_len), dtype=position_ids.dtype, device=position_ids.device)
                data.batch["position_ids"] = torch.cat([left, pad, right], dim=1)
            elif position_ids.dim() == 3:
                left = position_ids[:, :, :prompt_len]
                right = position_ids[:, :, prompt_len:]
                pad = torch.zeros(
                    (len(data), position_ids.shape[1], pad_len), dtype=position_ids.dtype, device=position_ids.device
                )
                data.batch["position_ids"] = torch.cat([left, pad, right], dim=2)
        if "routed_experts" in data.batch:
            routed_experts = data.batch["routed_experts"]
            if routed_experts.dim() >= 2 and routed_experts.shape[1] == prompt_len + data.batch["responses"].shape[1]:
                left = routed_experts[:, :prompt_len]
                right = routed_experts[:, prompt_len:]
                pad = torch.zeros(
                    (len(data), pad_len, *routed_experts.shape[2:]),
                    dtype=routed_experts.dtype,
                    device=routed_experts.device,
                )
                data.batch["routed_experts"] = torch.cat([left, pad, right], dim=1)
        return data

    def _pad_response_width_for_concat(self, data: DataProto, target_response_len: int) -> DataProto:
        response_len = data.batch["responses"].shape[1]
        pad_len = target_response_len - response_len
        if pad_len <= 0:
            return data
        pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        # igsd_topk_* intentionally stays at compact action width;
        # igsd_topk_positions maps it back to the padded response in FSDP.
        response_like_keys = {
            "responses",
            "response_mask",
            "old_log_probs",
            "ref_log_prob",
            "advantages",
            "returns",
            "values",
            "token_level_scores",
            "token_level_rewards",
            "rollout_log_probs",
            "rollout_is_weights",
            "igsd_policy_mask",
            "igsd_distill_weights",
            "igsd_query_token_mask",
            "igsd_query_distill_mask",
            "igsd_candidate_mask",
            "igsd_gate",
            "igsd_turn_weight",
            "igsd_ig_reward",
            "igsd_ig_student",
            "igsd_ig_teacher",
            "igsd_teacher_log_probs",
            "igsd_token_gain",
            "igsd_token_gate",
            "igsd_token_gate_valid",
            "igsd_token_delta_valid",
        }
        for key in response_like_keys & set(data.batch.keys()):
            tensor = data.batch[key]
            if tensor.dim() < 2 or tensor.shape[1] != response_len:
                continue
            fill = pad_id if key == "responses" else 0
            pad_shape = (len(data), pad_len, *tensor.shape[2:])
            pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=tensor.device)
            data.batch[key] = torch.cat([tensor, pad], dim=1)

        if "input_ids" in data.batch:
            tensor = data.batch["input_ids"]
            pad = torch.full((len(data), pad_len), pad_id, dtype=tensor.dtype, device=tensor.device)
            data.batch["input_ids"] = torch.cat([tensor, pad], dim=1)
        if "attention_mask" in data.batch:
            tensor = data.batch["attention_mask"]
            pad = torch.zeros((len(data), pad_len), dtype=tensor.dtype, device=tensor.device)
            data.batch["attention_mask"] = torch.cat([tensor, pad], dim=1)
        if "position_ids" in data.batch:
            position_ids = data.batch["position_ids"]
            if position_ids.dim() == 2:
                pad = torch.zeros((len(data), pad_len), dtype=position_ids.dtype, device=position_ids.device)
                data.batch["position_ids"] = torch.cat([position_ids, pad], dim=1)
            elif position_ids.dim() == 3:
                pad = torch.zeros(
                    (len(data), position_ids.shape[1], pad_len), dtype=position_ids.dtype, device=position_ids.device
                )
                data.batch["position_ids"] = torch.cat([position_ids, pad], dim=2)
        if "routed_experts" in data.batch:
            routed_experts = data.batch["routed_experts"]
            expected_seq_len = data.batch["prompts"].shape[1] + response_len
            if routed_experts.dim() >= 2 and routed_experts.shape[1] == expected_seq_len:
                pad = torch.zeros(
                    (len(data), pad_len, *routed_experts.shape[2:]),
                    dtype=routed_experts.dtype,
                    device=routed_experts.device,
                )
                data.batch["routed_experts"] = torch.cat([routed_experts, pad], dim=1)
        return data

    def _align_igsd_aux_for_concat(self, batch: DataProto, aux: DataProto) -> tuple[DataProto, DataProto]:
        target_prompt_len = max(batch.batch["prompts"].shape[1], aux.batch["prompts"].shape[1])
        target_response_len = max(batch.batch["responses"].shape[1], aux.batch["responses"].shape[1])
        batch = self._pad_prompt_width_for_concat(batch, target_prompt_len)
        aux = self._pad_prompt_width_for_concat(aux, target_prompt_len)
        batch = self._pad_response_width_for_concat(batch, target_response_len)
        aux = self._pad_response_width_for_concat(aux, target_response_len)

        all_non_tensor_keys = set(batch.non_tensor_batch.keys()) | set(aux.non_tensor_batch.keys())
        for key in all_non_tensor_keys:
            if key not in batch.non_tensor_batch:
                batch.non_tensor_batch[key] = np.array([None] * len(batch), dtype=object)
            if key not in aux.non_tensor_batch:
                aux.non_tensor_batch[key] = np.array([None] * len(aux), dtype=object)

        for key, tensor in list(batch.batch.items()):
            if key in aux.batch:
                continue
            shape = (len(aux), *tensor.shape[1:])
            aux.batch[key] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        for key, tensor in list(aux.batch.items()):
            if key in batch.batch:
                continue
            shape = (len(batch), *tensor.shape[1:])
            batch.batch[key] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

        mismatched = {
            key: (tuple(batch.batch[key].shape[1:]), tuple(aux.batch[key].shape[1:]))
            for key in batch.batch.keys()
            if key in aux.batch and tuple(batch.batch[key].shape[1:]) != tuple(aux.batch[key].shape[1:])
        }
        if mismatched:
            raise ValueError(f"IGSD batch tensor shapes remain incompatible after alignment: {mismatched}")

        aux.meta_info = batch.meta_info
        return batch, aux

    def _attach_igsd_teacher_aux(self, batch: DataProto, teacher_aux: DataProto) -> tuple[DataProto, dict[str, float]]:
        loss_agg_mode = str(self.config.actor_rollout_ref.actor.loss_agg_mode)
        if loss_agg_mode != "token-mean":
            raise NotImplementedError(
                "IGSD M2 BC currently requires actor.loss_agg_mode='token-mean' so PPO and BC "
                f"normalizers remain independent, got {loss_agg_mode!r}"
            )
        response_mask = batch.batch["response_mask"]
        batch.batch["igsd_policy_mask"] = torch.ones_like(response_mask, dtype=torch.bool)
        if "igsd_distill_weights" not in batch.batch:
            batch.batch["igsd_distill_weights"] = torch.zeros_like(response_mask, dtype=torch.float32)
        batch.non_tensor_batch["igsd_is_teacher_aux"] = np.array([False] * len(batch), dtype=object)

        teacher_response_mask = teacher_aux.batch["response_mask"]
        teacher_aux.batch["igsd_policy_mask"] = torch.zeros_like(teacher_response_mask, dtype=torch.bool)
        lambda_coef = float(self.config.algorithm.get("igsd_lambda", 0.1) or 0.0)
        gate_source = str(self.config.algorithm.get("igsd_gate_source", "paired_ig")).lower()
        likelihood_gate_metrics: dict[str, float] = {}

        # M2 batches already carry paired-IG gate weights. Keep a fallback for
        # non-M2 providers, but never overwrite a computed delta gate.
        if "igsd_distill_weights" not in teacher_aux.batch:
            teacher_aux.batch["igsd_gate"] = torch.ones_like(teacher_response_mask, dtype=torch.float32)
            teacher_aux.batch["igsd_distill_weights"] = teacher_response_mask.float() * lambda_coef

        # Use injected query mask if available, otherwise fall back to full response mask
        if "igsd_query_distill_mask" in teacher_aux.batch:
            pass  # M2 builds the canonical action mask with the auxiliary batch.
        else:
            teacher_aux.batch["igsd_query_distill_mask"] = teacher_response_mask.bool()
        teacher_aux.batch["igsd_candidate_mask"] = torch.zeros_like(teacher_response_mask, dtype=torch.bool)

        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob_padded(teacher_aux)
        old_log_prob.batch.pop("entropys", None)
        teacher_aux = teacher_aux.union(old_log_prob)
        if gate_source == "action_likelihood_gap":
            likelihood_gate_metrics = apply_action_likelihood_gate(
                teacher_aux,
                self.config.algorithm,
                global_step=self.global_steps,
                lambda_coef=lambda_coef,
            )
        if self.use_reference_policy:
            teacher_aux = teacher_aux.union(self._compute_ref_log_prob_padded(teacher_aux))

        for key in ("token_level_scores", "token_level_rewards", "advantages", "returns"):
            teacher_aux.batch[key] = torch.zeros_like(teacher_response_mask, dtype=torch.float32)
        if self.use_critic:
            teacher_aux.batch["values"] = torch.zeros_like(teacher_response_mask, dtype=torch.float32)
        # Ensure rollout_log_probs exists on teacher aux if the main batch has it,
        # so that rollout correction metrics don't crash on shape mismatch.
        if "rollout_log_probs" in batch.batch and "rollout_log_probs" not in teacher_aux.batch:
            teacher_aux.batch["rollout_log_probs"] = torch.zeros_like(teacher_response_mask, dtype=torch.float32)

        teacher_aux.non_tensor_batch["igsd_is_teacher_aux"] = np.array([True] * len(teacher_aux), dtype=object)

        # Preserve the baseline number of optimizer mini-batches. Expand each
        # mini-batch only to the nearest DP x micro-batch multiple needed to mix
        # the auxiliary samples into the PPO updates.
        base_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        base_mini_batch_size = base_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        if len(batch) % base_mini_batch_size != 0:
            raise ValueError(
                f"IGSD main batch size {len(batch)} is not divisible by baseline mini-batch "
                f"size {base_mini_batch_size}"
            )
        num_optimizer_mini_batches = max(len(batch) // base_mini_batch_size, 1)
        actor_world_size = max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1)
        use_dynamic_bsz = bool(self.config.actor_rollout_ref.actor.get("use_dynamic_bsz", False))
        actor_micro_batch_size = self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", None)
        if not use_dynamic_bsz and actor_micro_batch_size is None:
            raise NotImplementedError(
                "IGSD M2 requires actor.ppo_micro_batch_size_per_gpu when dynamic batching is disabled"
            )
        actor_micro_batch_size = 1 if use_dynamic_bsz else max(int(actor_micro_batch_size), 1)
        mini_batch_granularity = actor_world_size * actor_micro_batch_size
        combined_size = len(batch) + len(teacher_aux)
        real_teacher_aux_size = len(teacher_aux)
        total_granularity = num_optimizer_mini_batches * mini_batch_granularity
        target_size = ((combined_size + total_granularity - 1) // total_granularity) * total_granularity
        igsd_mini_batch_size = target_size // num_optimizer_mini_batches
        if igsd_mini_batch_size % mini_batch_granularity != 0:
            raise AssertionError(
                f"IGSD mini-batch size {igsd_mini_batch_size} is not divisible by actor DP x micro-batch "
                f"granularity {mini_batch_granularity}"
            )
        if combined_size != target_size:
            aux_pad_needed = target_size - combined_size
            # Replicate samples only for divisibility, then explicitly zero every
            # training weight on the copies. A shallow slice preserves the source
            # weights and would silently over-count BC targets.
            padding_protos = []
            remaining = aux_pad_needed
            while remaining > 0:
                take = min(remaining, len(teacher_aux))
                padding = deepcopy(teacher_aux[:take])
                for key in (
                    "igsd_distill_weights",
                    "igsd_query_token_mask",
                    "igsd_query_distill_mask",
                    "igsd_topk_mask",
                    "igsd_gate",
                    "igsd_turn_weight",
                    "igsd_token_gain",
                    "igsd_token_gate",
                    "igsd_token_gate_valid",
                    "igsd_token_delta_valid",
                    # Padding copies must not enter the globally reduced OPD
                    # row denominator. Real zero-gate rows remain eligible.
                    "igsd_opd_row_eligible",
                ):
                    if key in padding.batch:
                        padding.batch[key].zero_()
                padding.batch["igsd_policy_mask"].zero_()
                padding.batch["response_mask"].zero_()
                remaining -= take
                padding_protos.append(padding)
            teacher_aux = DataProto.concat([teacher_aux] + padding_protos)

        batch, teacher_aux = self._align_igsd_aux_for_concat(batch, teacher_aux)
        actor_batch = DataProto.concat([batch, teacher_aux])
        if len(actor_batch) != target_size:
            raise AssertionError(f"IGSD actor batch has size {len(actor_batch)}, expected {target_size}")

        actor_batch.meta_info["global_token_num"] = torch.sum(actor_batch.batch["attention_mask"], dim=-1).tolist()
        actor_batch.meta_info["igsd_mini_batch_size_override"] = igsd_mini_batch_size
        actor_batch.meta_info["igsd_shuffle_override"] = True
        metrics = {
            "igsd/teacher_old_log_prob_mfu": float(old_log_prob_mfu),
            "igsd/gate_source_is_paired_ig": float(gate_source == "paired_ig"),
            "igsd/gate_source_is_action_likelihood_gap": float(
                gate_source == "action_likelihood_gap"
            ),
            "igsd/actor_batch_with_teacher_aux": float(len(actor_batch)),
            "igsd/teacher_aux_real_count": float(real_teacher_aux_size),
            "igsd/teacher_aux_padding_count": float(len(teacher_aux) - real_teacher_aux_size),
            "igsd/teacher_aux_padding_frac": float(
                (len(teacher_aux) - real_teacher_aux_size) / max(len(teacher_aux), 1)
            ),
            "igsd/actor_mini_batch_size": float(igsd_mini_batch_size),
            "igsd/actor_optimizer_mini_batch_count": float(num_optimizer_mini_batches),
            "igsd/teacher_distill_token_frac": float(
                (teacher_aux.batch["igsd_distill_weights"] > 0).float().mean().detach().cpu().item()
            ),
            "igsd/teacher_distill_weight_sum": float(
                teacher_aux.batch["igsd_distill_weights"].float().sum().detach().cpu().item()
            ),
        }
        if "igsd_teacher_log_probs" in teacher_aux.batch:
            selected = teacher_aux.batch["igsd_distill_weights"] > 0
            metrics["igsd/teacher_soft_log_prob_token_count"] = float(selected.float().sum().detach().cpu().item())
            metrics["igsd/teacher_soft_log_prob_selected_mean"] = (
                float(teacher_aux.batch["igsd_teacher_log_probs"][selected].float().mean().detach().cpu().item())
                if bool(selected.any())
                else 0.0
            )
        metrics.update(likelihood_gate_metrics)
        return actor_batch, metrics

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        mini_batch_size_override = batch.meta_info.pop("igsd_mini_batch_size_override", None)
        shuffle_override = batch.meta_info.pop("igsd_shuffle_override", None)
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        if "igsd_policy_mask" in batch_td:
            # Engine-level token normalization must count PPO tokens only. BC is
            # normalized separately by igsd_distill_weights inside ppo_loss.
            batch_td["loss_mask"] = batch_td["response_mask"].bool() & batch_td["igsd_policy_mask"].bool()
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        igsd_distill_mode = str(self.config.algorithm.get("igsd_distill_mode", "bc")).lower()
        igsd_distill_target = str(self.config.algorithm.get("igsd_distill_target", "teacher_action")).lower()
        igsd_use_topk = (
            igsd_distill_mode in {"topk_reverse_kl", "topk_jsd", "candidate_pair_jsd"}
            and "igsd_topk_candidate_ids" in batch_td
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        if mini_batch_size_override is not None:
            ppo_mini_batch_size = int(mini_batch_size_override)
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        if shuffle_override is not None:
            shuffle = bool(shuffle_override)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            igsd_distill_mode=igsd_distill_mode,
            igsd_distill_target=igsd_distill_target,
            igsd_use_topk=igsd_use_topk,
            igsd_distill_topk=int(self.config.algorithm.get("igsd_distill_topk", 50) or 50),
            igsd_rkl_diagnostics_enabled=bool(
                self.config.algorithm.get("igsd_rkl_diagnostics_enabled", False)
            ),
            igsd_rkl_floor_log_prob=float(
                self.config.algorithm.get("igsd_rkl_floor_log_prob", -30.0)
            ),
            igsd_rkl_teacher_low_threshold=float(
                self.config.algorithm.get("igsd_rkl_teacher_low_threshold", -10.0)
            ),
            igsd_rkl_student_high_threshold=float(
                self.config.algorithm.get("igsd_rkl_student_high_threshold", -5.0)
            ),
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._log_rank0(
            "[driver] fit start: total_epochs=%s total_training_steps=%s train_batches=%s val_batches=%s async_rollout=%s",
            self.config.trainer.total_epochs,
            self.total_training_steps,
            len(self.train_dataloader),
            len(self.val_dataloader),
            self.async_rollout_mode,
        )

        # load checkpoint and update weights before doing anything
        self._log_rank0("[driver] fit: load checkpoint start")
        self._load_checkpoint()
        self._log_rank0("[driver] fit: load checkpoint done, global_steps=%s", self.global_steps)
        self._log_rank0("[driver] fit: update rollout weights start")
        self.checkpoint_manager.update_weights(self.global_steps)
        self._log_rank0("[driver] fit: update rollout weights done")

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            self._log_rank0("[driver] fit: val_before_train enabled, start initial validation")
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            self._log_rank0("[driver] epoch %s start", epoch)
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                self._log_rank0(
                    "[driver] step %s start: epoch=%s batch_size=%s rollout_n=%s",
                    self.global_steps,
                    epoch,
                    len(batch),
                    self.config.actor_rollout_ref.rollout.n,
                    level=logging.DEBUG,
                )

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        self._log_rank0(
                            "[driver] step %s generate start: repeated_batch=%s",
                            self.global_steps,
                            len(gen_batch_output),
                            level=logging.DEBUG,
                        )
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        self._log_rank0("[driver] step %s generate done", self.global_steps, level=logging.DEBUG)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        self._log_rank0("[driver] step %s reward start", self.global_steps, level=logging.DEBUG)
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    entropy_turn_records: dict[tuple[int, int], dict[str, float | int | None]] = {}
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            old_log_prob_metrics.update(
                                self._compute_search_turn_entropy_metrics(
                                    batch=batch,
                                    entropys=entropys,
                                    reward_tensor=reward_tensor,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    turn_records=entropy_turn_records,
                                )
                            )
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        run_legacy_igsd = self.config.algorithm.get("enable_igsd", False) and str(
                            self.config.algorithm.get("igsd_provider", "none")
                        ) != "hindsight_rollout"
                        if self.config.algorithm.get("enable_ig_reward", False) or run_legacy_igsd:
                            # Compute and inject IG_student before gate computation
                            ig_student_metrics = inject_ig_student_into_batch(
                                batch=batch,
                                tokenizer=self.tokenizer,
                                config=self.config.algorithm,
                                actor_rollout_wg=self.actor_rollout_wg,
                                log_prob_micro_batch_size_per_gpu=self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                            )
                            metrics.update(ig_student_metrics)

                            prepared = prepare_igsd_batch(
                                batch=batch, config=self.config.algorithm, global_step=self.global_steps
                            )
                            batch = prepared.batch
                            metrics.update(prepared.metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    igsd_teacher_aux_batch = None
                    if self.config.algorithm.get("enable_igsd", False) and str(
                        self.config.algorithm.get("igsd_provider", "none")
                    ) == "hindsight_rollout":
                        with marked_timer("igsd_teacher", timing_raw, color="purple"):
                            self._log_rank0(
                                "[driver] step %s IGSD hindsight teacher rollout start",
                                self.global_steps,
                                level=logging.DEBUG,
                            )
                            igsd_teacher_aux_batch, igsd_teacher_metrics = self._build_igsd_hindsight_teacher_batch(
                                batch
                            )
                            metrics.update(igsd_teacher_metrics)
                            self._log_rank0(
                                "[driver] step %s IGSD hindsight teacher rollout done: aux=%s",
                                self.global_steps,
                                0 if igsd_teacher_aux_batch is None else len(igsd_teacher_aux_batch),
                                level=logging.DEBUG,
                            )

                    # M2 converts the privileged one-query rollout into an
                    # unprivileged student-prefix BC sample and computes paired IG.
                    if igsd_teacher_aux_batch is not None:
                        with marked_timer("igsd_m2", timing_raw, color="purple"):
                            igsd_teacher_aux_batch, igsd_m2_metrics = self._build_igsd_m2_bc_batch(
                                igsd_teacher_aux_batch,
                                turn_entropy_records=entropy_turn_records,
                            )
                        metrics.update(igsd_m2_metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_batch = batch
                            if igsd_teacher_aux_batch is not None:
                                actor_batch, igsd_aux_metrics = self._attach_igsd_teacher_aux(
                                    batch, igsd_teacher_aux_batch
                                )
                                metrics.update(igsd_aux_metrics)
                            actor_output = self._update_actor(actor_batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Log extra reward scores separately for visibility in training metrics.
                if reward_extra_infos_dict:
                    for key, values in reward_extra_infos_dict.items():
                        arr = np.array(values, dtype=np.float32)
                        if arr.size == 0:
                            continue
                        # if key.startswith("first_") :
                        #     all_count = float(np.sum(arr))
                        #     metrics[f"reward_extra/{key}/prob"] = all_count / len(values)
                        #     continue
                        # metrics[f"reward_extra/{key}/mean"] = float(arr.mean())
                        # metrics[f"reward_extra/{key}/min"] = float(arr.min())
                        # metrics[f"reward_extra/{key}/max"] = float(arr.max())

                        if key.startswith("first_"):
                            metrics[f"reward_extra/{key}/prob"] = float(arr.mean())
                        else:
                            metrics[f"reward_extra/{key}/mean"] = float(arr.mean())
                            metrics[f"reward_extra/{key}/min"] = float(arr.min())
                            metrics[f"reward_extra/{key}/max"] = float(arr.max())
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
