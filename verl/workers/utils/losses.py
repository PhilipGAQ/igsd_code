# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import math

import torch
from tensordict import TensorDict

from verl.trainer.diffusion.diffusion_algos import kl_penalty_image
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.igsd_distill import (
    compute_candidate_pair_jsd,
    compute_event_reverse_kl_diagnostics,
    compute_topk_union_jsd,
    compute_topk_union_reverse_kl,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric, OptionalMetric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _compact_model_output(model_output: dict, key: str) -> torch.Tensor:
    value = model_output[key]
    return value.to_padded_tensor(0.0) if value.is_nested else value


def _add_tail_metrics(
    metrics: dict,
    prefix: str,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    low_quantiles: bool = False,
) -> None:
    selected = values.detach().float()[mask.bool()]
    if selected.numel() == 0:
        return
    if low_quantiles:
        metrics[f"{prefix}_p01_microbatch_mean"] = OptionalMetric(
            value=torch.quantile(selected, 0.01), aggregation=AggregationType.MEAN
        )
        metrics[f"{prefix}_p05_microbatch_mean"] = OptionalMetric(
            value=torch.quantile(selected, 0.05), aggregation=AggregationType.MEAN
        )
        metrics[f"{prefix}_min"] = OptionalMetric(value=selected.min(), aggregation=AggregationType.MIN)
    else:
        selected = selected.abs()
        metrics[f"{prefix}_p95_microbatch_mean"] = OptionalMetric(
            value=torch.quantile(selected, 0.95), aggregation=AggregationType.MEAN
        )
        metrics[f"{prefix}_p99_microbatch_mean"] = OptionalMetric(
            value=torch.quantile(selected, 0.99), aggregation=AggregationType.MEAN
        )
        metrics[f"{prefix}_max"] = OptionalMetric(value=selected.max(), aggregation=AggregationType.MAX)


def _masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (values.float() * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def _add_pair_correlation(
    metrics: dict,
    name: str,
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    first = first.detach().float()[mask]
    second = second.detach().float()[mask]
    if first.numel() < 2:
        return
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if denominator <= torch.finfo(first.dtype).eps:
        return
    metrics[f"{name}_microbatch_mean"] = OptionalMetric(
        value=(first * second).sum() / denominator,
        aggregation=AggregationType.MEAN,
    )


def ppo_loss(
    config: ActorConfig,
    model_output=None,
    data: TensorDict | None = None,
    dp_group=None,
    student_logits: torch.Tensor | None = None,
):
    """Compute PPO loss, or IGSD top-k outputs when called as a logits processor.

    FSDP invokes the logits-processor path before ``model_output`` exists, so
    both ``model_output`` and ``data`` must remain optional in this shared
    callback signature.
    """
    if student_logits is not None:
        distill_mode = str(tu.get_non_tensor_data(data=data, key="igsd_distill_mode", default="bc")).lower()
        topk = int(tu.get_non_tensor_data(data=data, key="igsd_distill_topk", default=50))
        compact_mask = data.get("igsd_topk_compact_mask", None)
        if compact_mask is None:
            teacher_log_probs = data["igsd_topk_teacher_log_probs"]
            candidate_ids = data["igsd_topk_candidate_ids"]
        else:
            compact_mask = compact_mask.bool()
            teacher_log_probs = data["igsd_topk_teacher_log_probs"][compact_mask].unsqueeze(0)
            candidate_ids = data["igsd_topk_candidate_ids"][compact_mask].unsqueeze(0)
        kwargs = {
            "student_logits": student_logits,
            "teacher_log_probs": teacher_log_probs,
            "candidate_ids": candidate_ids,
            "topk": topk,
        }
        if distill_mode == "candidate_pair_jsd":
            return compute_candidate_pair_jsd(
                student_logits=student_logits,
                teacher_log_probs=teacher_log_probs,
                candidate_ids=candidate_ids,
            )
        if distill_mode == "topk_reverse_kl":
            return compute_topk_union_reverse_kl(**kwargs)
        if distill_mode == "topk_jsd":
            return compute_topk_union_jsd(**kwargs)
        raise ValueError(
            "IGSD logits processor requires candidate_pair_jsd, topk_reverse_kl, or topk_jsd, "
            f"got {distill_mode!r}"
        )

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    igsd_bc_token_count = tu.get_non_tensor_data(data=data, key="igsd_bc_token_count", default=None)
    igsd_opd_row_count = tu.get_non_tensor_data(data=data, key="igsd_opd_row_count", default=None)
    igsd_dp_size = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    igsd_distill_mode = str(tu.get_non_tensor_data(data=data, key="igsd_distill_mode", default="bc")).lower()
    igsd_distill_target = str(
        tu.get_non_tensor_data(data=data, key="igsd_distill_target", default="teacher_action")
    ).lower()
    igsd_rkl_diagnostics_enabled = bool(
        tu.get_non_tensor_data(data=data, key="igsd_rkl_diagnostics_enabled", default=False)
    )
    igsd_rkl_floor_log_prob = float(
        tu.get_non_tensor_data(data=data, key="igsd_rkl_floor_log_prob", default=-30.0)
    )
    igsd_rkl_teacher_low_threshold = float(
        tu.get_non_tensor_data(data=data, key="igsd_rkl_teacher_low_threshold", default=-10.0)
    )
    igsd_rkl_student_high_threshold = float(
        tu.get_non_tensor_data(data=data, key="igsd_rkl_student_high_threshold", default=-5.0)
    )
    metrics["actor/igsd_rkl_diagnostics_enabled"] = float(igsd_rkl_diagnostics_enabled)

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    if "igsd_policy_mask" in data:
        fields.append("igsd_policy_mask")
    if "igsd_distill_weights" in data:
        fields.append("igsd_distill_weights")
    if "igsd_teacher_log_probs" in data:
        fields.append("igsd_teacher_log_probs")
    if "igsd_query_token_mask" in data:
        fields.append("igsd_query_token_mask")
    if "igsd_token_gate_valid" in data:
        # Prefix-intervention IG normalizes gates over all branch-valid query
        # positions. Keep this mask so hard gates do not change the loss
        # denominator merely by setting some accepted-token weights to zero.
        fields.append("igsd_token_gate_valid")
        if "igsd_query_distill_mask" in data:
            fields.append("igsd_query_distill_mask")
    if "igsd_opd_row_eligible" in data:
        fields.append("igsd_opd_row_eligible")
    for key in ("igsd_gate", "igsd_ig_student", "igsd_ig_teacher"):
        if key in data:
            fields.append(key)
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    if "igsd_policy_mask" in data:
        response_mask = response_mask & data["igsd_policy_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if "igsd_distill_weights" in data:
        distill_weights = data["igsd_distill_weights"].to(log_prob.dtype)
        selected_tokens = distill_weights > 0
        token_count = selected_tokens.float().sum()
        weight_sum = distill_weights.float().sum()
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
        if igsd_distill_mode not in supported_distill_modes:
            raise NotImplementedError(
                f"Unsupported IGSD distill mode {igsd_distill_mode!r}; "
                f"supported modes are {sorted(supported_distill_modes)}"
            )
        needs_teacher_log_probs = igsd_distill_mode in {
            "jsd",
            "event_jsd",
            "forward_kl",
            "reverse_kl",
            "event_reverse_kl",
            "confidence_bc",
        }
        if needs_teacher_log_probs and "igsd_teacher_log_probs" not in data:
            raise KeyError(f"IGSD distill mode {igsd_distill_mode!r} requires igsd_teacher_log_probs")
        effective_event_floor = (
            igsd_rkl_floor_log_prob
            if igsd_distill_mode in {"reverse_kl", "event_reverse_kl"}
            else -30.0
        )

        if bool(selected_tokens.any()):
            # Weighted token distillation loss divided by the number of selected
            # teacher-action tokens. distill_weights already includes igsd_lambda.
            safe_log_prob = torch.where(selected_tokens, log_prob, torch.zeros_like(log_prob))
            if igsd_distill_mode in {"bc", "nll"}:
                token_loss = -safe_log_prob
            elif igsd_distill_mode in {"topk_reverse_kl", "topk_jsd", "candidate_pair_jsd"}:
                loss_key = {
                    "topk_reverse_kl": "igsd_topk_reverse_kl",
                    "topk_jsd": "igsd_topk_jsd",
                    "candidate_pair_jsd": "igsd_candidate_pair_jsd",
                }[igsd_distill_mode]
                token_loss = _compact_model_output(model_output, loss_key).float()
                if token_loss.shape != selected_tokens.shape:
                    raise ValueError(
                        "IGSD top-k loss does not align with distillation mask: "
                        f"{token_loss.shape=}, {selected_tokens.shape=}"
                    )
                token_loss = torch.where(selected_tokens, token_loss, torch.zeros_like(token_loss))
            else:
                teacher_log_prob = data["igsd_teacher_log_probs"].float().detach()
                safe_student_log_prob = safe_log_prob.float()
                safe_teacher_log_prob = torch.where(
                    selected_tokens, teacher_log_prob, torch.zeros_like(teacher_log_prob)
                )
                clamped_teacher_log_prob = safe_teacher_log_prob.clamp(
                    min=effective_event_floor, max=0.0
                )
                clamped_student_log_prob = safe_student_log_prob.clamp(
                    min=effective_event_floor, max=0.0
                )
                teacher_prob = clamped_teacher_log_prob.exp()
                student_prob = clamped_student_log_prob.exp()

                if igsd_distill_mode in {"jsd", "event_jsd"}:
                    eps = torch.finfo(teacher_prob.dtype).eps
                    mixture_prob = (0.5 * (teacher_prob + student_prob)).clamp(min=eps, max=1.0 - eps)
                    teacher_prob = teacher_prob.clamp(min=eps, max=1.0 - eps)
                    student_prob = student_prob.clamp(min=eps, max=1.0 - eps)
                    teacher_log_event = teacher_prob.log()
                    student_log_event = student_prob.log()
                    mixture_log_prob = mixture_prob.log()
                    mixture_log_not_prob = (1.0 - mixture_prob).log()
                    token_loss = 0.5 * (
                        teacher_prob * (teacher_log_event - mixture_log_prob)
                        + (1.0 - teacher_prob) * ((1.0 - teacher_prob).log() - mixture_log_not_prob)
                        + student_prob * (student_log_event - mixture_log_prob)
                        + (1.0 - student_prob) * ((1.0 - student_prob).log() - mixture_log_not_prob)
                    )
                elif igsd_distill_mode == "forward_kl":
                    eps = torch.finfo(teacher_prob.dtype).eps
                    teacher_prob = teacher_prob.clamp(min=eps, max=1.0 - eps)
                    student_prob = student_prob.clamp(min=eps, max=1.0 - eps)
                    token_loss = teacher_prob * (teacher_prob.log() - student_prob.log()) + (
                        1.0 - teacher_prob
                    ) * ((1.0 - teacher_prob).log() - (1.0 - student_prob).log())
                elif igsd_distill_mode in {"reverse_kl", "event_reverse_kl"}:
                    eps = torch.finfo(teacher_prob.dtype).eps
                    teacher_prob = teacher_prob.clamp(min=eps, max=1.0 - eps)
                    student_prob = student_prob.clamp(min=eps, max=1.0 - eps)
                    token_loss = student_prob * (student_prob.log() - teacher_prob.log()) + (
                        1.0 - student_prob
                    ) * ((1.0 - student_prob).log() - (1.0 - teacher_prob).log())
                elif igsd_distill_mode == "confidence_bc":
                    token_loss = -safe_log_prob * teacher_prob
                else:
                    raise AssertionError(f"unhandled IGSD distill mode {igsd_distill_mode!r}")
                if igsd_distill_mode in {
                    "jsd",
                    "event_jsd",
                    "forward_kl",
                    "reverse_kl",
                    "event_reverse_kl",
                }:
                    token_loss = token_loss.clamp_min(0.0)
                token_loss = torch.where(selected_tokens, token_loss, torch.zeros_like(token_loss))

            # distill_weights already includes igsd_lambda. Legacy OPD divides
            # by positive-weight tokens; token-intervention modes divide by
            # their configured query denominator so sparse gates do not amplify
            # the remaining positions.
            weighted_token_loss = token_loss * distill_weights
            if igsd_distill_target == "student_on_policy":
                if "igsd_token_gate_valid" in data:
                    query_mask = data.get("igsd_query_token_mask", torch.zeros_like(selected_tokens)).bool()
                    denominator_mask = data["igsd_token_gate_valid"].bool() & query_mask
                    # The current G1 configuration uses query_only, but keep
                    # full-tool-call behavior well-defined for direct callers:
                    # structural tokens are always part of the distillation
                    # span and therefore count in the row denominator.
                    if "igsd_query_distill_mask" in data:
                        structural_mask = data["igsd_query_distill_mask"].bool() & ~query_mask
                        denominator_mask = denominator_mask | structural_mask
                    per_row_token_count = denominator_mask.sum(dim=-1)
                else:
                    per_row_token_count = selected_tokens.sum(dim=-1)
                valid_rows = selected_tokens.any(dim=-1)
                eligible_rows = (
                    data["igsd_opd_row_eligible"].bool()
                    if igsd_distill_target == "student_on_policy"
                    and "igsd_opd_row_eligible" in data
                    else valid_rows
                )
                per_row_loss = weighted_token_loss.sum(dim=-1) / per_row_token_count.clamp_min(1)
                if igsd_opd_row_count is None:
                    bc_denominator = eligible_rows.sum().clamp_min(1)
                    bc_scale = 1.0
                else:
                    bc_denominator = max(float(igsd_opd_row_count), 1.0)
                    bc_scale = float(igsd_dp_size)
                igsd_bc_loss = per_row_loss[eligible_rows].sum() / bc_denominator * bc_scale
            else:
                weighted_distill_loss = weighted_token_loss.sum()
                if igsd_bc_token_count is None:
                    bc_denominator = torch.clamp(token_count, min=1.0)
                    bc_scale = 1.0
                else:
                    bc_denominator = max(float(igsd_bc_token_count), 1.0)
                    bc_scale = float(igsd_dp_size)
                igsd_bc_loss = weighted_distill_loss / bc_denominator * bc_scale
            policy_loss = policy_loss + igsd_bc_loss
        else:
            igsd_bc_loss = log_prob.new_zeros(())
            token_loss = torch.zeros_like(log_prob)
            safe_log_prob = torch.zeros_like(log_prob)
        metrics["actor/igsd_bc_loss"] = Metric(value=igsd_bc_loss, aggregation=metric_aggregation)
        metrics["actor/igsd_distill_loss"] = Metric(value=igsd_bc_loss, aggregation=metric_aggregation)
        metrics["actor/igsd_bc_token_count"] = Metric(value=token_count, aggregation=AggregationType.SUM)
        metrics["actor/igsd_bc_weight_sum"] = Metric(value=weight_sum, aggregation=AggregationType.SUM)
        selected_row_count = (
            selected_tokens.any(dim=-1).sum()
            if igsd_distill_target == "student_on_policy"
            else token_count.new_zeros(())
        )
        eligible_row_count = selected_row_count
        if igsd_distill_target == "student_on_policy" and "igsd_opd_row_eligible" in data:
            # Strict G1 retains all-zero token-gate queries in the OPD row
            # denominator. Report that denominator rather than only the rows
            # that happen to have a positive token weight.
            eligible_row_count = data["igsd_opd_row_eligible"].bool().sum()
        metrics["actor/igsd_opd_row_count"] = Metric(
            value=eligible_row_count,
            aggregation=AggregationType.SUM,
        )
        metrics["actor/igsd_opd_selected_row_count"] = Metric(
            value=selected_row_count,
            aggregation=AggregationType.SUM,
        )
        metrics["actor/igsd_distill_target_is_student_on_policy"] = float(
            igsd_distill_target == "student_on_policy"
        )
        metrics["actor/igsd_distill_mode_is_jsd"] = float(
            igsd_distill_mode in {"jsd", "event_jsd", "topk_jsd", "candidate_pair_jsd"}
        )
        metrics["actor/igsd_distill_mode_is_forward_kl"] = float(igsd_distill_mode == "forward_kl")
        metrics["actor/igsd_distill_mode_is_reverse_kl"] = float(
            igsd_distill_mode in {"reverse_kl", "event_reverse_kl", "topk_reverse_kl"}
        )
        metrics["actor/igsd_distill_mode_is_topk_reverse_kl"] = float(
            igsd_distill_mode == "topk_reverse_kl"
        )
        metrics["actor/igsd_distill_mode_is_topk_jsd"] = float(igsd_distill_mode == "topk_jsd")
        metrics["actor/igsd_distill_mode_is_candidate_pair_jsd"] = float(
            igsd_distill_mode == "candidate_pair_jsd"
        )
        metrics["actor/igsd_distill_mode_is_confidence_bc"] = float(igsd_distill_mode == "confidence_bc")
        if igsd_distill_mode in {"reverse_kl", "event_reverse_kl"}:
            metrics["actor/igsd_rkl_floor_log_prob"] = igsd_rkl_floor_log_prob
            metrics["actor/igsd_rkl_effective_numeric_floor_log_prob"] = max(
                igsd_rkl_floor_log_prob,
                math.log(torch.finfo(torch.float32).eps),
            )
        # Mean NLL over distilled tokens (without lambda scaling) for monitoring learning progress.
        if bool(selected_tokens.any()):
            mean_nll = -(safe_log_prob[selected_tokens]).mean()
            metrics["actor/igsd_bc_mean_nll"] = Metric(value=mean_nll, aggregation=AggregationType.MEAN)
            metrics["actor/igsd_distill_token_loss_mean"] = Metric(
                value=token_loss[selected_tokens].mean(), aggregation=AggregationType.MEAN
            )
            if needs_teacher_log_probs:
                selected_teacher_log_prob = data["igsd_teacher_log_probs"].to(log_prob.dtype)[selected_tokens]
                selected_student_log_prob = log_prob[selected_tokens]
                metrics["actor/igsd_teacher_log_prob_mean"] = OptionalMetric(
                    value=selected_teacher_log_prob.mean(), aggregation=AggregationType.MEAN
                )
                metrics["actor/igsd_student_log_prob_on_teacher_mean"] = OptionalMetric(
                    value=selected_student_log_prob.mean(), aggregation=AggregationType.MEAN
                )
                metrics["actor/igsd_teacher_prob_mean"] = OptionalMetric(
                    value=selected_teacher_log_prob.clamp(min=effective_event_floor, max=0.0).exp().mean(),
                    aggregation=AggregationType.MEAN,
                )
                metrics["actor/igsd_student_prob_on_teacher_mean"] = OptionalMetric(
                    value=selected_student_log_prob.clamp(min=effective_event_floor, max=0.0).exp().mean(),
                    aggregation=AggregationType.MEAN,
                )
                if igsd_rkl_diagnostics_enabled and igsd_distill_mode in {
                    "reverse_kl",
                    "event_reverse_kl",
                }:
                    diagnostics = compute_event_reverse_kl_diagnostics(
                        teacher_log_probs=data["igsd_teacher_log_probs"],
                        student_log_probs=log_prob,
                        selected_mask=selected_tokens,
                        floor_log_prob=igsd_rkl_floor_log_prob,
                        teacher_low_threshold=igsd_rkl_teacher_low_threshold,
                        student_high_threshold=igsd_rkl_student_high_threshold,
                    )
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_rkl/teacher_log_prob",
                        diagnostics["teacher_log_prob"],
                        selected_tokens,
                        low_quantiles=True,
                    )
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_rkl/student_log_prob",
                        diagnostics["student_log_prob"],
                        selected_tokens,
                        low_quantiles=True,
                    )
                    for name in (
                        "log_prob_gap",
                        "event_log_odds_gap",
                        "target_logit_grad_factor",
                        "pre_clip_target_logit_grad_factor",
                        "event_reverse_kl",
                    ):
                        _add_tail_metrics(
                            metrics,
                            f"actor/igsd_rkl/{name}",
                            diagnostics[name],
                            selected_tokens,
                        )
                    for name in (
                        "teacher_floor_clipped",
                        "student_floor_clipped",
                        "teacher_probability_floor_clipped",
                        "student_probability_floor_clipped",
                        "student_probability_ceiling_clipped",
                        "student_gradient_clipped",
                        "teacher_low_student_high",
                        "student_low_teacher_high",
                    ):
                        metrics[f"actor/igsd_rkl/{name}_frac"] = OptionalMetric(
                            value=diagnostics[name][selected_tokens].float().mean(),
                            aggregation=AggregationType.MEAN,
                        )
                    pair_has_tokens = selected_tokens.any(dim=-1)
                    pair_grad_max = diagnostics["target_logit_grad_factor"].abs().amax(dim=-1)
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_rkl/pair_grad_max",
                        pair_grad_max,
                        pair_has_tokens,
                    )
                    weighted_grad_proxy = diagnostics["target_logit_grad_factor"].abs() * distill_weights.float()
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_rkl/weighted_grad_proxy",
                        weighted_grad_proxy,
                        selected_tokens,
                    )
                    pair_weighted_grad_max = weighted_grad_proxy.amax(dim=-1)
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_rkl/pair_weighted_grad_max",
                        pair_weighted_grad_max,
                        pair_has_tokens,
                    )
                    pair_loss_max = diagnostics["event_reverse_kl"].amax(dim=-1)
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_rkl/pair_loss_max",
                        pair_loss_max,
                        pair_has_tokens,
                    )
                    if all(key in data for key in ("igsd_gate", "igsd_ig_student", "igsd_ig_teacher")):
                        pair_gate = _masked_row_mean(data["igsd_gate"], selected_tokens)
                        pair_delta_ig = _masked_row_mean(
                            data["igsd_ig_teacher"] - data["igsd_ig_student"], selected_tokens
                        )
                        _add_pair_correlation(
                            metrics,
                            "actor/igsd_rkl/gate_vs_grad_risk_pearson",
                            pair_gate,
                            pair_grad_max,
                            pair_has_tokens,
                        )
                        _add_pair_correlation(
                            metrics,
                            "actor/igsd_rkl/delta_ig_vs_grad_risk_pearson",
                            pair_delta_ig,
                            pair_grad_max,
                            pair_has_tokens,
                        )
                    query_mask = data.get("igsd_query_token_mask", torch.zeros_like(selected_tokens)).bool()
                    structure_mask = selected_tokens & ~query_mask
                    query_mask = selected_tokens & query_mask
                    if bool(query_mask.any()):
                        metrics["actor/igsd_rkl/query_grad_factor_abs_mean"] = OptionalMetric(
                            value=diagnostics["target_logit_grad_factor"][query_mask].abs().mean().detach(),
                            aggregation=AggregationType.MEAN,
                        )
                    if bool(structure_mask.any()):
                        metrics["actor/igsd_rkl/structure_grad_factor_abs_mean"] = OptionalMetric(
                            value=diagnostics["target_logit_grad_factor"][structure_mask].abs().mean().detach(),
                            aggregation=AggregationType.MEAN,
                        )
            if igsd_distill_mode in {"topk_reverse_kl", "topk_jsd"}:
                topk_outputs = {
                    name: _compact_model_output(model_output, name).float()
                    for name in (
                        "igsd_topk_teacher_entropy",
                        "igsd_topk_student_entropy",
                        "igsd_topk_overlap",
                        "igsd_topk_top1_agreement",
                        "igsd_topk_union_size",
                        "igsd_topk_teacher_retained_mass",
                        "igsd_topk_student_retained_mass",
                        "igsd_topk_max_abs_log_prob_gap",
                    )
                }
                for name, values in topk_outputs.items():
                    metrics[f"actor/{name}_mean"] = OptionalMetric(
                        value=values[selected_tokens].mean().detach(), aggregation=AggregationType.MEAN
                    )
                if igsd_rkl_diagnostics_enabled:
                    divergence_name = "reverse_kl" if igsd_distill_mode == "topk_reverse_kl" else "jsd"
                    _add_tail_metrics(
                        metrics,
                        f"actor/igsd_topk/{divergence_name}",
                        token_loss,
                        selected_tokens,
                    )
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_topk/max_abs_log_prob_gap",
                        topk_outputs["igsd_topk_max_abs_log_prob_gap"],
                        selected_tokens,
                    )
                    for name in (
                        "igsd_topk_teacher_retained_mass",
                        "igsd_topk_student_retained_mass",
                    ):
                        _add_tail_metrics(
                            metrics,
                            f"actor/igsd_topk/{name.removeprefix('igsd_topk_')}",
                            topk_outputs[name],
                            selected_tokens,
                            low_quantiles=True,
                        )
                    pair_has_tokens = selected_tokens.any(dim=-1)
                    pair_loss_max = token_loss.amax(dim=-1)
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_topk/pair_loss_max",
                        pair_loss_max,
                        pair_has_tokens,
                    )
                    weighted_divergence = token_loss * distill_weights.float()
                    _add_tail_metrics(
                        metrics,
                        f"actor/igsd_topk/weighted_{divergence_name}",
                        weighted_divergence,
                        selected_tokens,
                    )
                    if all(key in data for key in ("igsd_gate", "igsd_ig_student", "igsd_ig_teacher")):
                        pair_gate = _masked_row_mean(data["igsd_gate"], selected_tokens)
                        pair_delta_ig = _masked_row_mean(
                            data["igsd_ig_teacher"] - data["igsd_ig_student"], selected_tokens
                        )
                        risk_name = "kl_risk" if igsd_distill_mode == "topk_reverse_kl" else "jsd"
                        _add_pair_correlation(
                            metrics,
                            f"actor/igsd_topk/gate_vs_{risk_name}_pearson",
                            pair_gate,
                            pair_loss_max,
                            pair_has_tokens,
                        )
                        _add_pair_correlation(
                            metrics,
                            f"actor/igsd_topk/delta_ig_vs_{risk_name}_pearson",
                            pair_delta_ig,
                            pair_loss_max,
                            pair_has_tokens,
                        )
            if igsd_distill_mode == "candidate_pair_jsd":
                pair_outputs = {
                    name: _compact_model_output(model_output, name).float()
                    for name in (
                        "igsd_candidate_pair_teacher_token_prob",
                        "igsd_candidate_pair_student_teacher_token_prob",
                        "igsd_candidate_pair_teacher_retained_mass",
                        "igsd_candidate_pair_student_retained_mass",
                        "igsd_candidate_pair_abs_jsd_grad_log_odds",
                        "igsd_candidate_pair_teacher_log_odds",
                        "igsd_candidate_pair_student_log_odds",
                        "igsd_candidate_pair_log_odds_gap",
                        "igsd_candidate_pair_teacher_entropy",
                        "igsd_candidate_pair_student_entropy",
                    )
                    if name in model_output
                }
                for name, values in pair_outputs.items():
                    metrics[f"actor/{name}_mean"] = OptionalMetric(
                        value=values[selected_tokens].mean().detach(), aggregation=AggregationType.MEAN
                    )
                _add_tail_metrics(
                    metrics,
                    "actor/igsd_candidate_pair/jsd",
                    token_loss,
                    selected_tokens,
                )
                if "igsd_candidate_pair_log_odds_gap" in pair_outputs:
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_candidate_pair/log_odds_gap",
                        pair_outputs["igsd_candidate_pair_log_odds_gap"],
                        selected_tokens,
                    )
                for name in (
                    "igsd_candidate_pair_teacher_retained_mass",
                    "igsd_candidate_pair_student_retained_mass",
                ):
                    if name in pair_outputs:
                        _add_tail_metrics(
                            metrics,
                            "actor/igsd_candidate_pair/"
                            f"{name.removeprefix('igsd_candidate_pair_')}",
                            pair_outputs[name],
                            selected_tokens,
                            low_quantiles=True,
                        )
                abs_grad_log_odds = pair_outputs.get(
                    "igsd_candidate_pair_abs_jsd_grad_log_odds"
                )
                if abs_grad_log_odds is not None:
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_candidate_pair/abs_jsd_grad_log_odds",
                        abs_grad_log_odds,
                        selected_tokens,
                    )
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_candidate_pair/abs_jsd_grad_log_odds_low_tail",
                        abs_grad_log_odds,
                        selected_tokens,
                        low_quantiles=True,
                    )
                    weighted_abs_grad_log_odds = abs_grad_log_odds * distill_weights.float()
                    metrics[
                        "actor/igsd_candidate_pair_weighted_abs_jsd_grad_log_odds_mean"
                    ] = OptionalMetric(
                        value=weighted_abs_grad_log_odds[selected_tokens].mean().detach(),
                        aggregation=AggregationType.MEAN,
                    )
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_candidate_pair/weighted_abs_jsd_grad_log_odds",
                        weighted_abs_grad_log_odds,
                        selected_tokens,
                    )
                    _add_tail_metrics(
                        metrics,
                        "actor/igsd_candidate_pair/weighted_abs_jsd_grad_log_odds_low_tail",
                        weighted_abs_grad_log_odds,
                        selected_tokens,
                        low_quantiles=True,
                    )
                    token_gain = data.get("igsd_token_gain", None)
                    if token_gain is not None and token_gain.shape == abs_grad_log_odds.shape:
                        _add_pair_correlation(
                            metrics,
                            "actor/igsd_candidate_pair/delta_ig_vs_abs_jsd_grad_log_odds_pearson",
                            token_gain,
                            abs_grad_log_odds,
                            selected_tokens,
                        )
                        _add_pair_correlation(
                            metrics,
                            "actor/igsd_candidate_pair/delta_ig_vs_weighted_abs_jsd_grad_log_odds_pearson",
                            token_gain,
                            weighted_abs_grad_log_odds,
                            selected_tokens,
                        )
                weighted_pair_jsd = token_loss * distill_weights.float()
                _add_tail_metrics(
                    metrics,
                    "actor/igsd_candidate_pair/weighted_jsd",
                    weighted_pair_jsd,
                    selected_tokens,
                )
        else:
            metrics["actor/igsd_bc_mean_nll"] = Metric(value=0.0, aggregation=AggregationType.MEAN)
            metrics["actor/igsd_distill_token_loss_mean"] = Metric(value=0.0, aggregation=AggregationType.MEAN)

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics


def diffusion_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "flow_grpo")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    return policy_loss, metrics
