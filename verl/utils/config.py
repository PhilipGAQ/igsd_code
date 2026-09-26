# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def _validate_igsd_config(algorithm: DictConfig) -> None:
    """Fail early on inconsistent IGSD gate settings."""

    if not bool(algorithm.get("enable_igsd", False)):
        return

    # The curated release exposes only the environment-verified main method.
    # Framework compatibility code may still recognize other enum values,
    # but they must not silently become alternative IGSD training recipes.
    main_method = {
        "igsd_provider": "hindsight_rollout",
        "igsd_distill_mode": "candidate_pair_jsd",
        "igsd_distill_target": "student_on_policy",
        "igsd_distill_span": "query_only",
        "igsd_token_weight_mode": "sampled_action_pair_ig",
        "igsd_sampled_pair_reference_mode": "greedy_pair",
        "igsd_sampled_pair_candidate_mode": "sampled_action",
        "igsd_sampled_pair_gate_source": "environment_ig",
        "igsd_sampled_pair_gate_granularity": "token",
        "igsd_token_gate_mode": "rectified_sigmoid",
        "igsd_token_gate_normalization": "none",
        "igsd_row_verification_mode": "bypass",
        "igsd_gate_mode": "none",
        "igsd_teacher_context_mode": "success_queries_score",
        "igsd_teacher_turn_selection": "last_search",
    }
    for key, expected in main_method.items():
        if str(algorithm.get(key, "")).lower() != expected:
            raise ValueError(f"This release requires algorithm.{key}={expected}")

    distill_mode = str(algorithm.get("igsd_distill_mode", "bc")).lower()
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
        raise ValueError(
            f"algorithm.igsd_distill_mode={distill_mode!r} is unsupported; "
            f"expected one of {sorted(supported_distill_modes)}"
        )

    distill_target = str(algorithm.get("igsd_distill_target", "teacher_action")).lower()
    supported_distill_targets = {"teacher_action", "student_on_policy"}
    if distill_target not in supported_distill_targets:
        raise ValueError(
            f"algorithm.igsd_distill_target={distill_target!r} is unsupported; "
            f"expected one of {sorted(supported_distill_targets)}"
        )

    row_verification_mode = str(
        algorithm.get("igsd_row_verification_mode", "paired")
    ).lower()
    supported_row_verification_modes = {"paired", "bypass"}
    if row_verification_mode not in supported_row_verification_modes:
        raise ValueError(
            "algorithm.igsd_row_verification_mode="
            f"{row_verification_mode!r} is unsupported; expected one of "
            f"{sorted(supported_row_verification_modes)}"
        )

    gate_source = str(algorithm.get("igsd_gate_source", "paired_ig")).lower()
    supported_gate_sources = {"paired_ig", "action_likelihood_gap"}
    if gate_source not in supported_gate_sources:
        raise ValueError(
            f"algorithm.igsd_gate_source={gate_source!r} is unsupported; "
            f"expected one of {sorted(supported_gate_sources)}"
        )

    token_weight_mode = str(algorithm.get("igsd_token_weight_mode", "uniform")).lower()
    supported_token_weight_modes = {
        "uniform",
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    }
    if token_weight_mode not in supported_token_weight_modes:
        raise ValueError(
            f"algorithm.igsd_token_weight_mode={token_weight_mode!r} is unsupported; "
            f"expected one of {sorted(supported_token_weight_modes)}"
        )

    sampled_pair_gate_source = str(
        algorithm.get("igsd_sampled_pair_gate_source", "environment_ig")
    ).lower()
    sampled_pair_gate_granularity = str(
        algorithm.get("igsd_sampled_pair_gate_granularity", "token")
    ).lower()

    distill_span = str(algorithm.get("igsd_distill_span", "full_tool_call")).lower()
    if distill_span not in {"full_tool_call", "query_only"}:
        raise ValueError(
            f"algorithm.igsd_distill_span={distill_span!r} is unsupported; "
            "expected 'full_tool_call' or 'query_only'"
        )
    if distill_mode in {"topk_reverse_kl", "topk_jsd", "candidate_pair_jsd"}:
        topk = int(algorithm.get("igsd_distill_topk", 50))
        if topk <= 0:
            raise ValueError(f"algorithm.igsd_distill_topk must be positive, got {topk}")
        materialization_chunk_size = int(
            algorithm.get("igsd_topk_materialization_chunk_size", 512)
        )
        if materialization_chunk_size < 0:
            raise ValueError(
                "algorithm.igsd_topk_materialization_chunk_size must be non-negative, "
                f"got {materialization_chunk_size}"
            )
        if distill_mode == "candidate_pair_jsd" and topk != 2:
            raise ValueError(
                "algorithm.igsd_distill_topk must equal 2 for "
                "algorithm.igsd_distill_mode='candidate_pair_jsd'; the candidate-pair "
                "objective has a fixed two-token support"
            )

    if gate_source == "action_likelihood_gap":
        supported_likelihood_distill_modes = {
            "jsd",
            "event_jsd",
            "forward_kl",
            "reverse_kl",
            "event_reverse_kl",
            "confidence_bc",
        }
        if distill_target != "teacher_action":
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' requires "
                "algorithm.igsd_distill_target='teacher_action'"
            )
        if str(algorithm.get("igsd_provider", "none")) != "hindsight_rollout":
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' requires "
                "algorithm.igsd_provider='hindsight_rollout'"
            )
        if row_verification_mode != "paired":
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' requires "
                "algorithm.igsd_row_verification_mode='paired' for matched candidate coverage"
            )
        if token_weight_mode != "uniform":
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' requires "
                "algorithm.igsd_token_weight_mode='uniform'"
            )
        if distill_mode not in supported_likelihood_distill_modes:
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' requires a gathered-token "
                f"teacher likelihood, got igsd_distill_mode={distill_mode!r}"
            )
        if not bool(algorithm.get("igsd_enable_ig_forward", False)):
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' requires "
                "algorithm.igsd_enable_ig_forward=True for matched paired coverage"
            )
        if str(algorithm.get("igsd_turn_routing_mode", "independent")).lower() != "independent":
            raise ValueError(
                "algorithm.igsd_gate_source='action_likelihood_gap' currently requires "
                "algorithm.igsd_turn_routing_mode='independent'"
            )

    if distill_target == "student_on_policy":
        supported_on_policy_modes = {"topk_jsd", "candidate_pair_jsd"}
        if token_weight_mode == "sampled_action_pair_ig":
            supported_on_policy_modes.add("topk_reverse_kl")
        if distill_mode not in supported_on_policy_modes:
            raise ValueError(
                "algorithm.igsd_distill_target='student_on_policy' currently requires "
                "algorithm.igsd_distill_mode to be 'topk_jsd' or 'candidate_pair_jsd'; "
                "'topk_reverse_kl' is additionally supported for sampled_action_pair_ig"
            )
        if distill_span != "query_only":
            raise ValueError(
                "algorithm.igsd_distill_target='student_on_policy' requires "
                "algorithm.igsd_distill_span='query_only'"
            )
        if str(algorithm.get("igsd_provider", "none")) != "hindsight_rollout":
            raise ValueError(
                "algorithm.igsd_distill_target='student_on_policy' requires "
                "algorithm.igsd_provider='hindsight_rollout'"
            )
        needs_ig_forward = (
            row_verification_mode == "paired"
            or token_weight_mode in {
                "prefix_intervention_ig",
                "budgeted_local_candidate_ig",
            }
            or (
                token_weight_mode == "sampled_action_pair_ig"
                and sampled_pair_gate_source == "environment_ig"
            )
        )
        if needs_ig_forward and not bool(algorithm.get("igsd_enable_ig_forward", False)):
            raise ValueError(
                "algorithm.igsd_distill_target='student_on_policy' requires "
                "algorithm.igsd_enable_ig_forward=True when row verification or "
                "token intervention IG is enabled"
            )

    if (
        token_weight_mode in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }
        and distill_target != "student_on_policy"
    ):
        raise ValueError(
            "algorithm.igsd_token_weight_mode token intervention requires "
            "algorithm.igsd_distill_target='student_on_policy'"
        )

    if token_weight_mode in {
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    }:
        token_gate_mode = str(algorithm.get("igsd_token_gate_mode", "sigmoid")).lower()
        supported_token_gate_modes = {"sigmoid", "rectified_sigmoid", "hard", "none"}
        if token_gate_mode not in supported_token_gate_modes:
            raise ValueError(
                f"algorithm.igsd_token_gate_mode={token_gate_mode!r} is unsupported; "
                f"expected one of {sorted(supported_token_gate_modes)}"
            )
        token_gate_beta = float(algorithm.get("igsd_token_gate_beta", 5.0))
        if not math.isfinite(token_gate_beta) or token_gate_beta <= 0.0:
            raise ValueError(
                "algorithm.igsd_token_gate_beta must be finite and positive, "
                f"got {token_gate_beta}"
            )
        token_gate_margin = float(algorithm.get("igsd_token_gate_margin", 0.0))
        if not math.isfinite(token_gate_margin):
            raise ValueError(
                f"algorithm.igsd_token_gate_margin must be finite, got {token_gate_margin}"
            )
        token_gate_normalization = str(
            algorithm.get("igsd_token_gate_normalization", "row_mean")
        ).lower()
        supported_token_gate_normalizations = {"row_mean", "none"}
        if token_gate_normalization not in supported_token_gate_normalizations:
            raise ValueError(
                "algorithm.igsd_token_gate_normalization="
                f"{token_gate_normalization!r} is unsupported; expected one of "
                f"{sorted(supported_token_gate_normalizations)}"
            )
        token_invalid_fallback_weight = float(
            algorithm.get("igsd_token_invalid_fallback_weight", 0.0)
        )
        if (
            not math.isfinite(token_invalid_fallback_weight)
            or token_invalid_fallback_weight < 0.0
            or token_invalid_fallback_weight > 1.0
        ):
            raise ValueError(
                "algorithm.igsd_token_invalid_fallback_weight must be finite and in [0, 1], "
                f"got {token_invalid_fallback_weight}"
            )
        token_intervention_max_new_tokens = int(
            algorithm.get("igsd_token_intervention_max_new_tokens", 0)
        )
        if token_intervention_max_new_tokens < 0:
            raise ValueError(
                "algorithm.igsd_token_intervention_max_new_tokens must be non-negative, "
                f"got {token_intervention_max_new_tokens}"
            )
        token_intervention_max_concurrency = int(
            algorithm.get("igsd_token_intervention_max_concurrency", 0)
        )
        if token_intervention_max_concurrency < 0:
            raise ValueError(
                "algorithm.igsd_token_intervention_max_concurrency must be non-negative, "
                f"got {token_intervention_max_concurrency}"
            )

    if token_weight_mode == "budgeted_local_candidate_ig":
        budget_ratio = float(algorithm.get("igsd_budgeted_candidate_budget_ratio", 1.0))
        if not math.isfinite(budget_ratio) or budget_ratio < 0.0 or budget_ratio > 1.0:
            raise ValueError(
                "algorithm.igsd_budgeted_candidate_budget_ratio must be finite and in [0, 1], "
                f"got {budget_ratio}"
            )
        ambiguity_margin = float(algorithm.get("igsd_budgeted_ambiguity_log_margin", 0.5))
        if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0.0:
            raise ValueError(
                "algorithm.igsd_budgeted_ambiguity_log_margin must be finite and non-negative, "
                f"got {ambiguity_margin}"
            )
        confirmation_mode = str(
            algorithm.get("igsd_budgeted_confirmation_mode", "soft")
        ).lower()
        supported_confirmation_modes = {"soft", "positive_only", "none"}
        if confirmation_mode not in supported_confirmation_modes:
            raise ValueError(
                "algorithm.igsd_budgeted_confirmation_mode must be one of "
                f"{sorted(supported_confirmation_modes)}, got {confirmation_mode!r}"
            )
        confirmation_beta = float(algorithm.get("igsd_budgeted_confirmation_beta", 5.0))
        if not math.isfinite(confirmation_beta) or confirmation_beta <= 0.0:
            raise ValueError(
                "algorithm.igsd_budgeted_confirmation_beta must be finite and positive, "
                f"got {confirmation_beta}"
            )
        confirmation_margin = float(algorithm.get("igsd_budgeted_confirmation_margin", 0.0))
        if not math.isfinite(confirmation_margin):
            raise ValueError(
                "algorithm.igsd_budgeted_confirmation_margin must be finite, "
                f"got {confirmation_margin}"
            )
        target_eta = float(algorithm.get("igsd_budgeted_target_eta", 1.0))
        if not math.isfinite(target_eta) or target_eta < 0.0:
            raise ValueError(
                "algorithm.igsd_budgeted_target_eta must be finite and non-negative, "
                f"got {target_eta}"
            )
        if token_gate_normalization != "none":
            raise ValueError(
                "algorithm.igsd_token_weight_mode='budgeted_local_candidate_ig' requires "
                "algorithm.igsd_token_gate_normalization='none'"
            )
        if token_invalid_fallback_weight != 0.0:
            raise ValueError(
                "algorithm.igsd_token_weight_mode='budgeted_local_candidate_ig' requires "
                "algorithm.igsd_token_invalid_fallback_weight=0"
            )
        if row_verification_mode != "bypass":
            raise ValueError(
                "algorithm.igsd_token_weight_mode='budgeted_local_candidate_ig' requires "
                "algorithm.igsd_row_verification_mode='bypass'"
            )
        if not bool(algorithm.get("igsd_preserve_zero_token_gate_rows", False)):
            raise ValueError(
                "algorithm.igsd_token_weight_mode='budgeted_local_candidate_ig' requires "
                "algorithm.igsd_preserve_zero_token_gate_rows=True"
            )

    if distill_mode == "candidate_pair_jsd" and token_weight_mode != "sampled_action_pair_ig":
        raise ValueError(
            "algorithm.igsd_distill_mode='candidate_pair_jsd' requires "
            "algorithm.igsd_token_weight_mode='sampled_action_pair_ig'"
        )

    if token_weight_mode == "sampled_action_pair_ig":
        supported_sampled_pair_gate_granularities = {"token", "query_mean"}
        if sampled_pair_gate_granularity not in supported_sampled_pair_gate_granularities:
            raise ValueError(
                "algorithm.igsd_sampled_pair_gate_granularity must be one of "
                f"{sorted(supported_sampled_pair_gate_granularities)}, got "
                f"{sampled_pair_gate_granularity!r}"
            )
        if (
            sampled_pair_gate_granularity == "query_mean"
            and sampled_pair_gate_source != "environment_ig"
        ):
            raise ValueError(
                "algorithm.igsd_sampled_pair_gate_granularity='query_mean' requires "
                "algorithm.igsd_sampled_pair_gate_source='environment_ig'"
            )
        supported_sampled_pair_gate_sources = {
            "environment_ig",
            "likelihood_gap",
            "constant",
        }
        if sampled_pair_gate_source not in supported_sampled_pair_gate_sources:
            raise ValueError(
                "algorithm.igsd_sampled_pair_gate_source must be one of "
                f"{sorted(supported_sampled_pair_gate_sources)}, got "
                f"{sampled_pair_gate_source!r}"
            )
        audit_all_eligible = bool(
            algorithm.get("igsd_sampled_pair_audit_all_eligible", False)
        )
        budget_ratio = float(algorithm.get("igsd_sampled_pair_budget_ratio", 1.0))
        max_budget_ratio = 2.0 if audit_all_eligible else 1.0
        if (
            not math.isfinite(budget_ratio)
            or budget_ratio < 0.0
            or budget_ratio > max_budget_ratio
        ):
            raise ValueError(
                "algorithm.igsd_sampled_pair_budget_ratio must be finite and in "
                f"[0, {max_budget_ratio:g}] when "
                f"igsd_sampled_pair_audit_all_eligible={audit_all_eligible}, got {budget_ratio}"
            )
        log_odds_epsilon = float(
            algorithm.get("igsd_sampled_pair_log_odds_epsilon", 1e-6)
        )
        if not math.isfinite(log_odds_epsilon) or log_odds_epsilon < 0.0:
            raise ValueError(
                "algorithm.igsd_sampled_pair_log_odds_epsilon must be finite and non-negative, "
                f"got {log_odds_epsilon}"
            )
        if distill_mode not in {"candidate_pair_jsd", "topk_jsd", "topk_reverse_kl"}:
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_distill_mode to be 'candidate_pair_jsd', 'topk_jsd', "
                "or 'topk_reverse_kl'"
            )
        reference_mode = str(
            algorithm.get("igsd_sampled_pair_reference_mode", "greedy_pair")
        ).lower()
        if reference_mode not in {"greedy_pair", "fixed_s"}:
            raise ValueError(
                "algorithm.igsd_sampled_pair_reference_mode must be 'greedy_pair' or 'fixed_s', "
                f"got {reference_mode!r}"
            )
        candidate_mode = str(
            algorithm.get("igsd_sampled_pair_candidate_mode", "sampled_action")
        ).lower()
        if candidate_mode not in {"sampled_action", "student_top1"}:
            raise ValueError(
                "algorithm.igsd_sampled_pair_candidate_mode must be 'sampled_action' or "
                f"'student_top1', got {candidate_mode!r}"
            )
        if candidate_mode == "student_top1" and reference_mode != "greedy_pair":
            raise ValueError(
                "algorithm.igsd_sampled_pair_candidate_mode='student_top1' requires "
                "algorithm.igsd_sampled_pair_reference_mode='greedy_pair'"
            )
        if sampled_pair_gate_source == "likelihood_gap" and candidate_mode != "sampled_action":
            raise ValueError(
                "algorithm.igsd_sampled_pair_gate_source='likelihood_gap' requires "
                "algorithm.igsd_sampled_pair_candidate_mode='sampled_action' so the "
                "gate is log p_teacher(s_i)-log p_student(s_i)"
            )
        if (
            sampled_pair_gate_source == "environment_ig"
            and not bool(algorithm.get("igsd_enable_ig_forward", False))
        ):
            raise ValueError(
                "algorithm.igsd_sampled_pair_gate_source='environment_ig' requires "
                "algorithm.igsd_enable_ig_forward=True"
            )
        if token_gate_mode not in {"rectified_sigmoid", "hard"}:
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_token_gate_mode to be 'rectified_sigmoid' or 'hard'"
            )
        if token_gate_normalization != "none":
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_token_gate_normalization='none'"
            )
        if token_gate_margin < 0.0:
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_token_gate_margin >= 0 for positive-only pair gating"
            )
        if token_invalid_fallback_weight != 0.0:
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_token_invalid_fallback_weight=0"
            )
        if row_verification_mode != "bypass":
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_row_verification_mode='bypass'"
            )
        if not bool(algorithm.get("igsd_preserve_zero_token_gate_rows", False)):
            raise ValueError(
                "algorithm.igsd_token_weight_mode='sampled_action_pair_ig' requires "
                "algorithm.igsd_preserve_zero_token_gate_rows=True"
            )

    preserve_zero_token_gate_rows = bool(
        algorithm.get("igsd_preserve_zero_token_gate_rows", False)
    )
    if preserve_zero_token_gate_rows:
        if token_weight_mode not in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }:
            raise ValueError(
                "algorithm.igsd_preserve_zero_token_gate_rows=True requires "
                "a token-intervention igsd_token_weight_mode"
            )
        if row_verification_mode != "bypass":
            raise ValueError(
                "algorithm.igsd_preserve_zero_token_gate_rows=True requires "
                "algorithm.igsd_row_verification_mode='bypass'"
            )

    gate_mode = str(algorithm.get("igsd_gate_mode", "sigmoid")).lower()
    supported_gate_modes = {"sigmoid", "positive_sigmoid", "hard", "none", "lcb_sigmoid"}
    if gate_mode not in supported_gate_modes:
        raise ValueError(
            f"algorithm.igsd_gate_mode={gate_mode!r} is unsupported; "
            f"expected one of {sorted(supported_gate_modes)}"
        )
    if gate_source == "action_likelihood_gap" and gate_mode != "positive_sigmoid":
        raise ValueError(
            "algorithm.igsd_gate_source='action_likelihood_gap' requires "
            "algorithm.igsd_gate_mode='positive_sigmoid'"
        )

    if row_verification_mode == "bypass":
        if distill_target != "student_on_policy":
            raise ValueError(
                "algorithm.igsd_row_verification_mode='bypass' requires "
                "algorithm.igsd_distill_target='student_on_policy'"
            )
        if gate_mode != "none":
            raise ValueError(
                "algorithm.igsd_row_verification_mode='bypass' requires "
                "algorithm.igsd_gate_mode='none'"
            )
        turn_routing_mode = str(
            algorithm.get("igsd_turn_routing_mode", "independent")
        ).lower()
        if turn_routing_mode != "independent":
            raise ValueError(
                "algorithm.igsd_row_verification_mode='bypass' requires "
                "algorithm.igsd_turn_routing_mode='independent'"
            )

    beta = float(algorithm.get("igsd_beta", 5.0))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"algorithm.igsd_beta must be finite and positive, got {beta}")

    final_margin = float(algorithm.get("igsd_gate_margin", 0.0))
    initial_margin = float(algorithm.get("igsd_gate_margin_initial", 0.0))
    if not math.isfinite(final_margin):
        raise ValueError(f"algorithm.igsd_gate_margin must be finite, got {final_margin}")
    if not math.isfinite(initial_margin):
        raise ValueError(f"algorithm.igsd_gate_margin_initial must be finite, got {initial_margin}")

    margin_schedule = str(algorithm.get("igsd_gate_margin_schedule", "constant")).lower()
    supported_margin_schedules = {"constant", "early", "linear"}
    if margin_schedule not in supported_margin_schedules:
        raise ValueError(
            f"algorithm.igsd_gate_margin_schedule={margin_schedule!r} is unsupported; "
            f"expected one of {sorted(supported_margin_schedules)}"
        )
    margin_schedule_steps = int(algorithm.get("igsd_gate_margin_schedule_steps", 0))
    if margin_schedule_steps < 0:
        raise ValueError(
            "algorithm.igsd_gate_margin_schedule_steps must be non-negative, "
            f"got {margin_schedule_steps}"
        )
    if margin_schedule == "early" and margin_schedule_steps == 0:
        raise ValueError(
            "algorithm.igsd_gate_margin_schedule_steps must be positive when "
            "algorithm.igsd_gate_margin_schedule='early'"
        )
    if margin_schedule == "linear" and margin_schedule_steps < 2:
        raise ValueError(
            "algorithm.igsd_gate_margin_schedule_steps must be at least 2 when "
            "algorithm.igsd_gate_margin_schedule='linear'"
        )

    warmup_steps = int(algorithm.get("igsd_warmup_steps", 0))
    if warmup_steps < 0:
        raise ValueError(f"algorithm.igsd_warmup_steps must be non-negative, got {warmup_steps}")
    warmup_mode = str(algorithm.get("igsd_warmup_mode", "disable_distill")).lower()
    supported_warmup_modes = {"disable_distill", "ungated", "beta_ramp"}
    if warmup_mode not in supported_warmup_modes:
        raise ValueError(
            f"algorithm.igsd_warmup_mode={warmup_mode!r} is unsupported; "
            f"expected one of {sorted(supported_warmup_modes)}"
        )

    if gate_mode != "lcb_sigmoid":
        return

    if str(algorithm.get("igsd_provider", "none")) != "hindsight_rollout":
        raise ValueError(
            "algorithm.igsd_gate_mode='lcb_sigmoid' requires "
            "algorithm.igsd_provider='hindsight_rollout'"
        )
    if not bool(algorithm.get("igsd_enable_ig_forward", False)):
        raise ValueError(
            "algorithm.igsd_gate_mode='lcb_sigmoid' requires "
            "algorithm.igsd_enable_ig_forward=True"
        )
    num_counterfactual = int(algorithm.get("igsd_num_counterfactual", 0))
    min_counterfactual = int(algorithm.get("igsd_lcb_min_counterfactuals", 3))
    if min_counterfactual < 2:
        raise ValueError(
            "algorithm.igsd_lcb_min_counterfactuals must be at least 2 to estimate variance, "
            f"got {min_counterfactual}"
        )
    if num_counterfactual < min_counterfactual:
        raise ValueError(
            "algorithm.igsd_num_counterfactual must be >= "
            "algorithm.igsd_lcb_min_counterfactuals for the LCB gate, got "
            f"{num_counterfactual} < {min_counterfactual}"
        )
    lcb_kappa = float(algorithm.get("igsd_lcb_kappa", 1.0))
    if not math.isfinite(lcb_kappa) or lcb_kappa < 0.0:
        raise ValueError(
            f"algorithm.igsd_lcb_kappa must be finite and non-negative, got {lcb_kappa}"
        )


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    _validate_igsd_config(config.algorithm)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
