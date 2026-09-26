# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import math

import torch


def compute_topk_union_reverse_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    topk: int,
) -> dict[str, torch.Tensor]:
    """Compute categorical reverse KL on the teacher/student top-k union.

    ``candidate_ids`` contains the teacher top-k followed by the student top-k
    measured before the PPO update. ``teacher_log_probs`` contains full-vocab
    teacher log probabilities for every candidate. Both distributions are
    renormalized on the de-duplicated union support before computing
    ``KL(student || teacher)``.
    """

    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must have shape [batch, response, vocab], got {student_logits.shape}")
    if teacher_log_probs.shape != candidate_ids.shape:
        raise ValueError(
            "teacher_log_probs and candidate_ids must have identical shapes, got "
            f"{teacher_log_probs.shape} and {candidate_ids.shape}"
        )
    if student_logits.shape[:2] != candidate_ids.shape[:2]:
        raise ValueError(
            "IGSD top-k targets do not align with student logits: "
            f"{student_logits.shape[:2]=}, {candidate_ids.shape[:2]=}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if candidate_ids.shape[-1] != 2 * topk:
        raise ValueError(f"Expected {2 * topk} union candidates, got {candidate_ids.shape[-1]}")
    if candidate_ids.numel() and (candidate_ids.min() < 0 or candidate_ids.max() >= student_logits.shape[-1]):
        raise ValueError("candidate_ids contain token IDs outside the student vocabulary")

    candidate_ids = candidate_ids.long()
    teacher_log_probs = teacher_log_probs.float().detach()
    student_logits = student_logits.float()
    student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_ids)

    # Keep the first occurrence so overlap between teacher/student top-k does
    # not double count probability mass after support renormalization.
    equality = candidate_ids.unsqueeze(-1).eq(candidate_ids.unsqueeze(-2))
    support_width = candidate_ids.shape[-1]
    earlier = torch.tril(
        torch.ones((support_width, support_width), dtype=torch.bool, device=candidate_ids.device),
        diagonal=-1,
    )
    keep = ~(equality & earlier).any(dim=-1)
    negative_large = torch.finfo(student_candidate_logits.dtype).min

    teacher_masked = teacher_log_probs.masked_fill(~keep, negative_large)
    student_masked = student_candidate_logits.masked_fill(~keep, negative_large)
    teacher_log_z = torch.logsumexp(teacher_masked, dim=-1, keepdim=True)
    student_log_z = torch.logsumexp(student_masked, dim=-1, keepdim=True)
    teacher_log_p = teacher_masked - teacher_log_z
    student_log_p = student_masked - student_log_z
    teacher_p = torch.where(keep, teacher_log_p.exp(), torch.zeros_like(teacher_log_p))
    student_p = torch.where(keep, student_log_p.exp(), torch.zeros_like(student_log_p))

    reverse_kl_terms = torch.where(
        keep,
        student_p * (student_log_p - teacher_log_p),
        torch.zeros_like(student_p),
    )
    reverse_kl = reverse_kl_terms.sum(dim=-1).clamp_min(0.0)
    teacher_entropy = -(teacher_p * torch.where(keep, teacher_log_p, torch.zeros_like(teacher_log_p))).sum(dim=-1)
    student_entropy = -(student_p * torch.where(keep, student_log_p, torch.zeros_like(student_log_p))).sum(dim=-1)

    teacher_ids = candidate_ids[..., :topk]
    student_ids = candidate_ids[..., topk:]
    topk_overlap = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    top1_agreement = teacher_ids[..., 0].eq(student_ids[..., 0]).float()
    union_size = keep.float().sum(dim=-1)

    # Teacher inputs are full-vocabulary log probabilities. Student retained
    # mass can be computed from the current full-vocabulary logits before union
    # renormalization.
    teacher_retained_mass = torch.where(keep, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs)).sum(dim=-1)
    student_full_log_z = torch.logsumexp(student_logits, dim=-1)
    student_retained_mass = (student_log_z.squeeze(-1) - student_full_log_z).exp()
    log_prob_gap = torch.where(keep, student_log_p - teacher_log_p, torch.zeros_like(student_log_p))
    max_abs_log_prob_gap = log_prob_gap.abs().amax(dim=-1)

    return {
        "igsd_topk_reverse_kl": reverse_kl,
        "igsd_topk_teacher_entropy": teacher_entropy,
        "igsd_topk_student_entropy": student_entropy,
        "igsd_topk_overlap": topk_overlap,
        "igsd_topk_top1_agreement": top1_agreement,
        "igsd_topk_union_size": union_size,
        "igsd_topk_teacher_retained_mass": teacher_retained_mass,
        "igsd_topk_student_retained_mass": student_retained_mass,
        "igsd_topk_max_abs_log_prob_gap": max_abs_log_prob_gap,
    }


def compute_topk_union_jsd(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    topk: int,
) -> dict[str, torch.Tensor]:
    """Compute categorical JSD on the de-duplicated teacher/student top-k union."""

    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must have shape [batch, response, vocab], got {student_logits.shape}")
    if teacher_log_probs.shape != candidate_ids.shape:
        raise ValueError(
            "teacher_log_probs and candidate_ids must have identical shapes, got "
            f"{teacher_log_probs.shape} and {candidate_ids.shape}"
        )
    if student_logits.shape[:2] != candidate_ids.shape[:2]:
        raise ValueError(
            "IGSD top-k targets do not align with student logits: "
            f"{student_logits.shape[:2]=}, {candidate_ids.shape[:2]=}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if candidate_ids.shape[-1] != 2 * topk:
        raise ValueError(f"Expected {2 * topk} union candidates, got {candidate_ids.shape[-1]}")
    if candidate_ids.numel() and (candidate_ids.min() < 0 or candidate_ids.max() >= student_logits.shape[-1]):
        raise ValueError("candidate_ids contain token IDs outside the student vocabulary")

    candidate_ids = candidate_ids.long()
    teacher_log_probs = teacher_log_probs.float().detach()
    student_logits = student_logits.float()
    student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_ids)

    equality = candidate_ids.unsqueeze(-1).eq(candidate_ids.unsqueeze(-2))
    support_width = candidate_ids.shape[-1]
    earlier = torch.tril(
        torch.ones((support_width, support_width), dtype=torch.bool, device=candidate_ids.device),
        diagonal=-1,
    )
    keep = ~(equality & earlier).any(dim=-1)
    negative_large = torch.finfo(student_candidate_logits.dtype).min

    teacher_masked = teacher_log_probs.masked_fill(~keep, negative_large)
    student_masked = student_candidate_logits.masked_fill(~keep, negative_large)
    teacher_log_z = torch.logsumexp(teacher_masked, dim=-1, keepdim=True)
    student_log_z = torch.logsumexp(student_masked, dim=-1, keepdim=True)
    teacher_log_p = teacher_masked - teacher_log_z
    student_log_p = student_masked - student_log_z
    teacher_p = torch.where(keep, teacher_log_p.exp(), torch.zeros_like(teacher_log_p))
    student_p = torch.where(keep, student_log_p.exp(), torch.zeros_like(student_log_p))
    log_mixture = torch.logaddexp(teacher_log_p, student_log_p) - math.log(2.0)

    teacher_terms = torch.where(
        keep,
        teacher_p * (teacher_log_p - log_mixture),
        torch.zeros_like(teacher_p),
    )
    student_terms = torch.where(
        keep,
        student_p * (student_log_p - log_mixture),
        torch.zeros_like(student_p),
    )
    jsd = (0.5 * (teacher_terms.sum(dim=-1) + student_terms.sum(dim=-1))).clamp_min(0.0)
    teacher_entropy = -(teacher_p * torch.where(keep, teacher_log_p, torch.zeros_like(teacher_log_p))).sum(dim=-1)
    student_entropy = -(student_p * torch.where(keep, student_log_p, torch.zeros_like(student_log_p))).sum(dim=-1)

    teacher_ids = candidate_ids[..., :topk]
    student_ids = candidate_ids[..., topk:]
    topk_overlap = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    top1_agreement = teacher_ids[..., 0].eq(student_ids[..., 0]).float()
    union_size = keep.float().sum(dim=-1)
    teacher_retained_mass = torch.where(keep, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs)).sum(dim=-1)
    student_full_log_z = torch.logsumexp(student_logits, dim=-1)
    student_retained_mass = (student_log_z.squeeze(-1) - student_full_log_z).exp()
    log_prob_gap = torch.where(keep, student_log_p - teacher_log_p, torch.zeros_like(student_log_p))

    return {
        "igsd_topk_jsd": jsd,
        "igsd_topk_teacher_entropy": teacher_entropy,
        "igsd_topk_student_entropy": student_entropy,
        "igsd_topk_overlap": topk_overlap,
        "igsd_topk_top1_agreement": top1_agreement,
        "igsd_topk_union_size": union_size,
        "igsd_topk_teacher_retained_mass": teacher_retained_mass,
        "igsd_topk_student_retained_mass": student_retained_mass,
        "igsd_topk_max_abs_log_prob_gap": log_prob_gap.abs().amax(dim=-1),
    }


def _validate_local_candidate_inputs(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    candidate_width: int,
) -> None:
    if student_logits.ndim != 3:
        raise ValueError(
            f"student_logits must have shape [batch, response, vocab], got {student_logits.shape}"
        )
    expected_shape = (*student_logits.shape[:2], candidate_width)
    if teacher_log_probs.shape != expected_shape or candidate_ids.shape != expected_shape:
        raise ValueError(
            "teacher_log_probs and candidate_ids must have shape "
            f"{expected_shape}, got {teacher_log_probs.shape} and {candidate_ids.shape}"
        )
    if candidate_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError(f"candidate_ids must use an integer dtype, got {candidate_ids.dtype}")
    if candidate_ids.numel() and (
        candidate_ids.min().item() < 0 or candidate_ids.max().item() >= student_logits.shape[-1]
    ):
        raise ValueError("candidate_ids contain token IDs outside the student vocabulary")
    if not bool(torch.isfinite(teacher_log_probs).all()):
        raise ValueError("teacher_log_probs must contain only finite values")


def compute_candidate_pair_jsd(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute JSD on a verified two-token teacher/counterfactual pair.

    The first candidate is the teacher proposal and the second is the
    student-side reference action. Both distributions are renormalized on
    exactly this two-token support, so gradients with respect to vocabulary
    logits are zero outside the candidate pair. Callers must remove inactive
    positions before invoking this helper; duplicate IDs are rejected rather
    than silently turning an inactive row into a one-token loss. Returned
    retained-mass and analytic-gradient diagnostics are detached from the loss.
    """

    _validate_local_candidate_inputs(
        student_logits,
        teacher_log_probs,
        candidate_ids,
        candidate_width=2,
    )
    if candidate_ids.numel() and bool(candidate_ids[..., 0].eq(candidate_ids[..., 1]).any()):
        raise ValueError("candidate-pair JSD requires two distinct candidate token IDs")

    candidate_ids = candidate_ids.long()
    teacher_pair_logits = teacher_log_probs.float().detach()
    student_pair_logits = torch.gather(student_logits.float(), dim=-1, index=candidate_ids)
    teacher_log_p = torch.log_softmax(teacher_pair_logits, dim=-1)
    student_log_p = torch.log_softmax(student_pair_logits, dim=-1)
    teacher_p = teacher_log_p.exp()
    student_p = student_log_p.exp()
    log_mixture = torch.logaddexp(teacher_log_p, student_log_p) - math.log(2.0)
    jsd = 0.5 * (
        (teacher_p * (teacher_log_p - log_mixture)).sum(dim=-1)
        + (student_p * (student_log_p - log_mixture)).sum(dim=-1)
    )
    jsd = jsd.clamp_min(0.0)

    teacher_entropy = -(teacher_p * teacher_log_p).sum(dim=-1)
    student_entropy = -(student_p * student_log_p).sum(dim=-1)
    teacher_log_odds = teacher_log_p[..., 0] - teacher_log_p[..., 1]
    student_log_odds = student_log_p[..., 0] - student_log_p[..., 1]
    with torch.no_grad():
        teacher_retained_mass = torch.logsumexp(teacher_pair_logits, dim=-1).exp()
        student_log_normalizer = torch.logsumexp(student_logits.detach().float(), dim=-1)
        student_retained_mass = (
            torch.logsumexp(student_pair_logits.detach(), dim=-1) - student_log_normalizer
        ).exp()
        mixture_log_odds = log_mixture[..., 0] - log_mixture[..., 1]
        # For q=sigmoid(b), d JSD(p, q) / d b equals
        # 0.5*q*(1-q)*(logit(q)-logit((p+q)/2)).
        abs_jsd_grad_log_odds = (
            0.5
            * student_p[..., 0].detach()
            * student_p[..., 1].detach()
            * (student_log_odds.detach() - mixture_log_odds.detach())
        ).abs()
    return {
        "igsd_candidate_pair_jsd": jsd,
        "igsd_candidate_pair_teacher_token_prob": teacher_p[..., 0].detach(),
        "igsd_candidate_pair_student_teacher_token_prob": student_p[..., 0].detach(),
        "igsd_candidate_pair_teacher_retained_mass": teacher_retained_mass,
        "igsd_candidate_pair_student_retained_mass": student_retained_mass,
        "igsd_candidate_pair_abs_jsd_grad_log_odds": abs_jsd_grad_log_odds,
        "igsd_candidate_pair_teacher_log_odds": teacher_log_odds.detach(),
        "igsd_candidate_pair_student_log_odds": student_log_odds.detach(),
        "igsd_candidate_pair_log_odds_gap": (teacher_log_odds - student_log_odds).detach(),
        "igsd_candidate_pair_teacher_entropy": teacher_entropy.detach(),
        "igsd_candidate_pair_student_entropy": student_entropy.detach(),
    }


def compute_event_reverse_kl_diagnostics(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    floor_log_prob: float = -30.0,
    teacher_low_threshold: float = -10.0,
    student_high_threshold: float = -5.0,
) -> dict[str, torch.Tensor]:
    """Return per-token event-RKL stability signals without reducing them."""

    if teacher_log_probs.shape != student_log_probs.shape or selected_mask.shape != student_log_probs.shape:
        raise ValueError(
            "teacher_log_probs, student_log_probs, and selected_mask must align, got "
            f"{teacher_log_probs.shape}, {student_log_probs.shape}, {selected_mask.shape}"
        )

    teacher_raw = teacher_log_probs.float().detach()
    student_raw = student_log_probs.float()
    eps = torch.finfo(torch.float32).eps
    teacher_prob_before_numeric_clip = teacher_raw.clamp(min=floor_log_prob, max=0.0).exp()
    student_prob_before_numeric_clip = student_raw.clamp(min=floor_log_prob, max=0.0).exp()
    teacher_prob = teacher_prob_before_numeric_clip.clamp(min=eps, max=1.0 - eps)
    student_prob = student_prob_before_numeric_clip.clamp(min=eps, max=1.0 - eps)
    teacher_logit = torch.logit(teacher_prob)
    student_logit = torch.logit(student_prob)
    event_log_odds_gap = student_logit - teacher_logit
    pre_clip_target_logit_grad_factor = student_prob * (1.0 - student_prob) * event_log_odds_gap
    student_floor_clipped = student_raw.lt(floor_log_prob)
    student_probability_floor_clipped = student_prob_before_numeric_clip.lt(eps)
    student_probability_ceiling_clipped = student_prob_before_numeric_clip.gt(1.0 - eps)
    student_gradient_clipped = (
        student_floor_clipped | student_probability_floor_clipped | student_probability_ceiling_clipped
    )
    target_logit_grad_factor = torch.where(
        student_gradient_clipped,
        torch.zeros_like(pre_clip_target_logit_grad_factor),
        pre_clip_target_logit_grad_factor,
    )
    event_reverse_kl = student_prob * (student_prob.log() - teacher_prob.log()) + (
        1.0 - student_prob
    ) * ((1.0 - student_prob).log() - (1.0 - teacher_prob).log())

    selected_mask = selected_mask.bool()
    zeros = torch.zeros_like(student_prob)
    return {
        "teacher_log_prob": torch.where(selected_mask, teacher_raw, zeros),
        "student_log_prob": torch.where(selected_mask, student_raw, zeros),
        "log_prob_gap": torch.where(selected_mask, student_raw - teacher_raw, zeros),
        "event_log_odds_gap": torch.where(selected_mask, event_log_odds_gap, zeros),
        "target_logit_grad_factor": torch.where(selected_mask, target_logit_grad_factor, zeros),
        "pre_clip_target_logit_grad_factor": torch.where(
            selected_mask, pre_clip_target_logit_grad_factor, zeros
        ),
        "event_reverse_kl": torch.where(selected_mask, event_reverse_kl, zeros),
        "teacher_floor_clipped": selected_mask & teacher_raw.lt(floor_log_prob),
        "student_floor_clipped": selected_mask & student_floor_clipped,
        "teacher_probability_floor_clipped": selected_mask & teacher_prob_before_numeric_clip.lt(eps),
        "student_probability_floor_clipped": selected_mask & student_probability_floor_clipped,
        "student_probability_ceiling_clipped": selected_mask & student_probability_ceiling_clipped,
        "student_gradient_clipped": selected_mask & student_gradient_clipped,
        "teacher_low_student_high": selected_mask
        & teacher_raw.lt(teacher_low_threshold)
        & student_raw.gt(student_high_threshold),
        "student_low_teacher_high": selected_mask
        & student_raw.lt(teacher_low_threshold)
        & teacher_raw.gt(student_high_threshold),
    }
