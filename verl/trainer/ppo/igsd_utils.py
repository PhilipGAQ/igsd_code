# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Utilities for IG-gated self-distillation experiments.

The first IGSD training pass keeps teacher-query generation and counterfactual
IG computation behind a provider boundary. This module prepares masks, gates,
and logging tensors without changing the vanilla GRPO path when provider fields
are absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto


@dataclass
class IGSDPreparedBatch:
    batch: DataProto
    metrics: dict[str, float]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _safe_mean(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if tensor.numel() == 0:
        return 0.0
    values = tensor
    if mask is not None:
        values = values[mask.bool()]
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().detach().cpu().item())


def generated_span_end_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """Return a mask on the last token of every generated-token span.

    Search-R1 multi-turn traces use response_mask=1 for assistant-generated
    spans and response_mask=0 for tool responses. This mask is a cheap turn
    boundary proxy that works without text re-tokenization.
    """

    bool_mask = response_mask.bool()
    next_is_zero = torch.ones_like(bool_mask)
    if bool_mask.shape[-1] > 1:
        next_is_zero[..., :-1] = ~bool_mask[..., 1:]
    return bool_mask & next_is_zero


def non_final_generated_span_end_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """Return generated span ends except the final generated span per sample."""

    span_ends = generated_span_end_mask(response_mask)
    result = span_ends.clone()
    for row in range(span_ends.shape[0]):
        idx = torch.nonzero(span_ends[row], as_tuple=False).flatten()
        if idx.numel() > 0:
            result[row, idx[-1]] = False
    return result


def success_mask_from_scores(batch: DataProto) -> torch.Tensor:
    """Infer rollout success from reward/origin-score style fields."""

    if "origin_score" in batch.non_tensor_batch:
        values = np.asarray(batch.non_tensor_batch["origin_score"], dtype=np.float32)
        return torch.tensor(values > 0.0, device=batch.batch["responses"].device)
    return torch.zeros(len(batch), dtype=torch.bool, device=batch.batch["responses"].device)


def prompt_has_success_sibling(batch: DataProto) -> torch.Tensor:
    """Mark samples whose prompt group has at least one successful sibling."""

    success = success_mask_from_scores(batch).detach().cpu().numpy().astype(bool)
    uids = batch.non_tensor_batch.get("uid")
    if uids is None:
        return torch.zeros(len(batch), dtype=torch.bool, device=batch.batch["responses"].device)
    uid_to_success: dict[Any, bool] = {}
    for uid, ok in zip(uids, success, strict=False):
        uid_to_success[uid] = uid_to_success.get(uid, False) or bool(ok)
    values = [uid_to_success.get(uid, False) for uid in uids]
    return torch.tensor(values, dtype=torch.bool, device=batch.batch["responses"].device)


def _tensor_from_batch_field(batch: DataProto, key: str, like: torch.Tensor) -> torch.Tensor | None:
    if key in batch.batch:
        value = batch.batch[key].to(device=like.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value[:, None].expand_as(like)
        return value
    if key in batch.non_tensor_batch:
        value = torch.tensor(np.asarray(batch.non_tensor_batch[key]), device=like.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value[:, None].expand_as(like)
        return value
    return None


def _mask_from_batch_field(batch: DataProto, key: str, like: torch.Tensor) -> torch.Tensor | None:
    value = _tensor_from_batch_field(batch, key, like)
    if value is None:
        return None
    return value.bool()


def effective_gate_margin(config: Any, global_step: int) -> float:
    """Return the configured gate margin for the current optimizer step."""

    schedule = str(_cfg_get(config, "igsd_gate_margin_schedule", "constant")).lower()
    final_margin = float(_cfg_get(config, "igsd_gate_margin", 0.0) or 0.0)
    if not math.isfinite(final_margin):
        raise ValueError(f"algorithm.igsd_gate_margin must be finite, got {final_margin}")
    if schedule == "constant":
        return final_margin
    if schedule not in {"early", "linear"}:
        raise ValueError(
            "Unsupported IGSD gate margin schedule: "
            f"{schedule!r}; expected one of ['constant', 'early', 'linear']"
        )

    schedule_steps = int(_cfg_get(config, "igsd_gate_margin_schedule_steps", 0) or 0)
    if schedule_steps <= 0:
        raise ValueError(
            "algorithm.igsd_gate_margin_schedule_steps must be positive when "
            f"algorithm.igsd_gate_margin_schedule={schedule!r}"
        )
    initial_margin = float(_cfg_get(config, "igsd_gate_margin_initial", final_margin) or 0.0)
    if not math.isfinite(initial_margin):
        raise ValueError(f"algorithm.igsd_gate_margin_initial must be finite, got {initial_margin}")
    if schedule == "early":
        return initial_margin if int(global_step) <= schedule_steps else final_margin
    if schedule_steps < 2:
        raise ValueError(
            "algorithm.igsd_gate_margin_schedule_steps must be at least 2 when "
            "algorithm.igsd_gate_margin_schedule='linear'"
        )
    progress = min(max((int(global_step) - 1) / (schedule_steps - 1), 0.0), 1.0)
    return initial_margin + progress * (final_margin - initial_margin)


def compute_gate(
    delta: torch.Tensor,
    config: Any,
    global_step: int,
    delta_standard_error: torch.Tensor | None = None,
) -> torch.Tensor:
    warmup_steps = int(_cfg_get(config, "igsd_warmup_steps", 0) or 0)
    warmup_mode = str(_cfg_get(config, "igsd_warmup_mode", "disable_distill"))
    gate_mode = str(_cfg_get(config, "igsd_gate_mode", "sigmoid")).lower()
    beta = float(_cfg_get(config, "igsd_beta", 5.0) or 5.0)
    margin = effective_gate_margin(config, global_step)

    if warmup_steps > 0 and global_step <= warmup_steps:
        if warmup_mode == "disable_distill":
            return torch.zeros_like(delta)
        if warmup_mode == "ungated":
            return torch.ones_like(delta)
        if warmup_mode == "beta_ramp":
            beta *= max(float(global_step) / float(warmup_steps), 1e-6)

    if gate_mode == "none":
        return torch.ones_like(delta)
    if gate_mode == "hard":
        return (delta > margin).float()
    if gate_mode == "positive_sigmoid":
        gate = torch.sigmoid(beta * (delta - margin))
        return gate * (delta > margin).to(dtype=gate.dtype)
    if gate_mode == "lcb_sigmoid":
        if delta_standard_error is None:
            raise ValueError("IGSD lcb_sigmoid gate requires a paired delta standard error")
        if delta_standard_error.shape != delta.shape:
            raise ValueError(
                "IGSD delta standard error shape must match delta shape, got "
                f"{tuple(delta_standard_error.shape)} vs {tuple(delta.shape)}"
            )
        if not bool(torch.isfinite(delta_standard_error).all()):
            raise ValueError("IGSD delta standard error must contain only finite values")
        if bool((delta_standard_error < 0).any()):
            raise ValueError("IGSD delta standard error must be non-negative")
        kappa = float(_cfg_get(config, "igsd_lcb_kappa", 1.0) or 0.0)
        if not math.isfinite(kappa) or kappa < 0.0:
            raise ValueError(f"algorithm.igsd_lcb_kappa must be finite and non-negative, got {kappa}")
        delta = delta - kappa * delta_standard_error.to(device=delta.device, dtype=delta.dtype)
    elif gate_mode != "sigmoid":
        raise ValueError(f"Unsupported IGSD gate mode: {gate_mode}")
    return torch.sigmoid(beta * (delta - margin))


def prepare_igsd_batch(batch: DataProto, config: Any, global_step: int) -> IGSDPreparedBatch:
    """Attach IGSD masks/weights and return metrics for logging."""

    response_mask = batch.batch["response_mask"]
    device = response_mask.device
    turn_end_mask = non_final_generated_span_end_mask(response_mask)
    sibling_mask = prompt_has_success_sibling(batch).to(device=device)
    candidate_mask = turn_end_mask & sibling_mask[:, None]

    ig_student = _tensor_from_batch_field(batch, "igsd_ig_student", response_mask)
    ig_teacher = _tensor_from_batch_field(batch, "igsd_ig_teacher", response_mask)
    provider_missing = ig_student is None or ig_teacher is None

    if provider_missing:
        if bool(_cfg_get(config, "igsd_fail_on_missing_provider", False)):
            raise RuntimeError(
                "IGSD is enabled but igsd_ig_student/igsd_ig_teacher provider tensors are missing. "
                "Set algorithm.igsd_fail_on_missing_provider=False for metrics-only scaffolding."
            )
        ig_student = torch.zeros_like(response_mask, dtype=torch.float32)
        ig_teacher = torch.zeros_like(response_mask, dtype=torch.float32)

    delta = ig_teacher - ig_student
    delta_standard_error = _tensor_from_batch_field(batch, "igsd_ig_delta_standard_error", response_mask)
    if provider_missing:
        gate = torch.zeros_like(delta)
    else:
        gate = compute_gate(
            delta=delta,
            config=config,
            global_step=global_step,
            delta_standard_error=delta_standard_error,
        )
    if bool(_cfg_get(config, "igsd_detach_gate", True)):
        gate = gate.detach()

    provider_distill_weights = _tensor_from_batch_field(batch, "igsd_distill_weights", response_mask)
    query_distill_mask = _mask_from_batch_field(batch, "igsd_query_distill_mask", response_mask)
    if query_distill_mask is None:
        if provider_distill_weights is not None:
            query_distill_mask = provider_distill_weights != 0
        else:
            query_distill_mask = torch.zeros_like(candidate_mask)

    lambda_value = _cfg_get(config, "igsd_lambda", 0.1)
    lambda_coef = 0.1 if lambda_value is None else float(lambda_value)
    if provider_distill_weights is not None:
        distill_weights = provider_distill_weights.float() * lambda_coef
    else:
        sample_gate = (gate * candidate_mask.float()).max(dim=-1, keepdim=True).values
        distill_weights = sample_gate * query_distill_mask.float() * lambda_coef
    batch.batch["igsd_candidate_mask"] = candidate_mask
    batch.batch["igsd_query_distill_mask"] = query_distill_mask
    batch.batch["igsd_gate"] = gate.float()
    batch.batch["igsd_distill_weights"] = distill_weights.float()

    alpha = float(_cfg_get(config, "igsd_alpha", 0.1) or 0.0)
    if bool(_cfg_get(config, "enable_ig_reward", False)):
        ig_reward = ig_student * candidate_mask.float() * alpha
        batch.batch["token_level_scores"] = batch.batch["token_level_scores"] + ig_reward.to(
            batch.batch["token_level_scores"].dtype
        )
        batch.batch["igsd_ig_reward"] = ig_reward.float()

    effective_margin = effective_gate_margin(config, global_step)
    hard_open = delta > effective_margin
    valid_provider_mask = candidate_mask if not provider_missing else torch.zeros_like(candidate_mask)
    metrics = {
        "igsd/provider_missing": float(provider_missing),
        "igsd/success_sibling_frac": _safe_mean(sibling_mask.float()),
        "igsd/no_success_sibling_frac": _safe_mean((~sibling_mask).float()),
        "igsd/eligible_prompt_frac": _safe_mean((candidate_mask.sum(dim=-1) > 0).float()),
        "igsd/generated_turns_mean": _safe_mean(turn_end_mask.float().sum(dim=-1)),
        "igsd/candidate_turns_mean": _safe_mean(candidate_mask.float().sum(dim=-1)),
        "igsd/eligible_turn_frac": _safe_mean(candidate_mask.float(), turn_end_mask),
        "igsd/query_distill_token_frac": _safe_mean(query_distill_mask.float(), response_mask.bool()),
        "igsd/distill_token_frac": _safe_mean((distill_weights > 0).float(), response_mask.bool()),
        "igsd/distill_weight_sum": float(distill_weights.float().sum().detach().cpu().item()),
        "igsd/soft_gate_mean": _safe_mean(gate, valid_provider_mask),
        "igsd/effective_gate_margin": effective_margin,
        "igsd/gate_zero_frac": _safe_mean((gate <= 1e-6).float(), valid_provider_mask),
        "igsd/gate_mid_frac": _safe_mean(((gate > 1e-6) & (gate < 0.5)).float(), valid_provider_mask),
        "igsd/gate_high_frac": _safe_mean((gate >= 0.5).float(), valid_provider_mask),
        "igsd/hard_open_ratio": _safe_mean(hard_open.float(), valid_provider_mask),
        "igsd/bad_teacher_ratio": _safe_mean((delta <= 0).float(), valid_provider_mask),
        "igsd/ig_delta_pos_frac": _safe_mean((delta > 0).float(), valid_provider_mask),
        "igsd/ig_student_mean": _safe_mean(ig_student, valid_provider_mask),
        "igsd/ig_teacher_mean": _safe_mean(ig_teacher, valid_provider_mask),
        "igsd/ig_delta_mean": _safe_mean(delta, valid_provider_mask),
        "igsd/effective_distill_weight": _safe_mean(distill_weights, candidate_mask),
    }
    if "igsd_ig_reward" in batch.batch:
        metrics["igsd/ig_reward_mean"] = _safe_mean(batch.batch["igsd_ig_reward"], candidate_mask)
        metrics["igsd/ig_reward_sum"] = float(batch.batch["igsd_ig_reward"].float().sum().detach().cpu().item())
        metrics["igsd/ig_reward_nonzero_frac"] = _safe_mean(
            (batch.batch["igsd_ig_reward"] != 0).float(), candidate_mask
        )
    return IGSDPreparedBatch(batch=batch, metrics=metrics)
