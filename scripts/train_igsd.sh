#!/usr/bin/env bash
# Main IGSD training setup; requires a separately running compatible retriever.
set -euo pipefail

: "${IGSD_MODEL_PATH:?Set IGSD_MODEL_PATH to your model directory}"
: "${IGSD_TRAIN_DATA:?Set IGSD_TRAIN_DATA to your training Parquet file}"
: "${IGSD_VAL_DATA:?Set IGSD_VAL_DATA to your validation Parquet file}"
: "${IGSD_OUTPUT_DIR:?Set IGSD_OUTPUT_DIR to a writable output directory}"
: "${IGSD_RETRIEVAL_URL:?Set IGSD_RETRIEVAL_URL to the local /retrieve endpoint}"

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TOOL_CONFIG="${ROOT}/examples/search_agent_rl/config/tool_config/search_tool_config.yaml"
for path in "${IGSD_MODEL_PATH}" "${IGSD_TRAIN_DATA}" "${IGSD_VAL_DATA}" "${TOOL_CONFIG}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'Missing required path: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ "${IGSD_RETRIEVAL_URL}" != http://127.0.0.1:* && "${IGSD_RETRIEVAL_URL}" != http://localhost:* ]]; then
  printf 'The training launcher expects a local retrieval endpoint; set up a local port forward if needed.\n' >&2
  exit 2
fi

if [[ -n "${IGSD_TOOL_CONFIG:-}" ]]; then
  TOOL_CONFIG="${IGSD_TOOL_CONFIG}"
elif [[ "${IGSD_RETRIEVAL_URL}" != 'http://127.0.0.1:8000/retrieve' ]]; then
  printf 'Set IGSD_TOOL_CONFIG to a matching local search-tool YAML.\n' >&2
  exit 2
fi
if [[ ! -f "${TOOL_CONFIG}" ]]; then
  printf 'Missing search-tool configuration: %s\n' "${TOOL_CONFIG}" >&2
  exit 2
fi

mkdir -p "${IGSD_OUTPUT_DIR}"
cd "${ROOT}"
python3 -m verl.trainer.main_ppo \
  --config-path="${ROOT}/examples/search_agent_rl/config" \
  --config-name=search_multiturn_grpo \
  algorithm.adv_estimator=grpo \
  algorithm.enable_ig_reward=False \
  algorithm.enable_igsd=True \
  algorithm.igsd_provider=hindsight_rollout \
  algorithm.igsd_distill_mode=candidate_pair_jsd \
  algorithm.igsd_distill_target=student_on_policy \
  algorithm.igsd_distill_topk=2 \
  algorithm.igsd_distill_span=query_only \
  algorithm.igsd_gate_mode=none \
  algorithm.igsd_row_verification_mode=bypass \
  algorithm.igsd_token_weight_mode=sampled_action_pair_ig \
  algorithm.igsd_token_gate_mode=rectified_sigmoid \
  algorithm.igsd_token_gate_beta=5.0 \
  algorithm.igsd_token_gate_margin=0.0 \
  algorithm.igsd_token_gate_normalization=none \
  algorithm.igsd_token_invalid_fallback_weight=0.0 \
  algorithm.igsd_preserve_zero_token_gate_rows=True \
  algorithm.igsd_enable_ig_forward=True \
  algorithm.igsd_sampled_pair_reference_mode=greedy_pair \
  algorithm.igsd_sampled_pair_gate_source=environment_ig \
  algorithm.igsd_sampled_pair_gate_granularity=token \
  algorithm.igsd_sampled_pair_budget_ratio=1.0 \
  algorithm.igsd_sampled_pair_log_odds_epsilon=1e-6 \
  algorithm.igsd_sampled_pair_candidate_mode=sampled_action \
  algorithm.igsd_teacher_context_mode=success_queries_score \
  algorithm.igsd_teacher_turn_selection=last_search \
  algorithm.igsd_max_teacher_turns_per_prompt=1 \
  algorithm.igsd_num_counterfactual=3 \
  algorithm.igsd_max_gt_aliases=3 \
  algorithm.igsd_max_pseudo_batch=256 \
  algorithm.igsd_max_pseudo_seq_len=15000 \
  algorithm.igsd_lambda=0.01 \
  algorithm.igsd_token_intervention_max_new_tokens=256 \
  algorithm.igsd_token_intervention_max_concurrency=8 \
  algorithm.igsd_retrieval_url="${IGSD_RETRIEVAL_URL}" \
  algorithm.igsd_retrieval_topk=3 \
  distillation.enabled=False \
  data.train_files="${IGSD_TRAIN_DATA}" \
  data.val_files="${IGSD_VAL_DATA}" \
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_prompt_length=4096 \
  data.max_response_length=3000 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${IGSD_MODEL_PATH}" \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.max_model_len=15000 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1500 \
  actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary=False \
  actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call=1 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.total_training_steps=400 \
  trainer.total_epochs=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.resume_mode=disable \
  trainer.val_before_train=False \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=igsd \
  trainer.experiment_name="${IGSD_RUN_NAME:-igsd_main}" \
  trainer.n_gpus_per_node="${IGSD_N_GPUS_PER_NODE:-8}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${IGSD_OUTPUT_DIR}" \
  +data.apply_chat_template_kwargs.enable_thinking=False
