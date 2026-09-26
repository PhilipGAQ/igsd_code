# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integration layer for IG computation in the training loop.

This module provides the bridge between the driver-level training loop
(ray_trainer.py) and the IG computation modules (igsd_ig_compute.py,
igsd_ig_forward.py, igsd_teacher_ig.py).

Two modes of IG computation:

1. **Structural (heuristic)** — default, no model forward pass needed:
     IG_student = 1.0 for turns that have retrieved documents
     IG_student = 0.0 for turns without documents

2. **Model-based (forward pass)** — enabled by igsd_enable_ig_forward=True:
     IG_t = mean_k log π(a*_k | C_real,t) - (1/N) Σ_j mean_k log π(a*_k | C_rand,j,t)
     Uses actor_rollout_wg.compute_log_prob() for batched forward pass.
     Measures actual causal contribution of retrieved docs to answer generation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.igsd_ig_compute import (
    SearchTurn,
    build_query_span_mask,
    parse_trajectory_tokens,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Helper: get ground-truth answers from batch
# ---------------------------------------------------------------------------


def _get_answer_texts(batch: DataProto, tokenizer: Any) -> list[str]:
    """Extract ground-truth answers for each sample in the batch.

    Looks in several places:
    1. non_tensor_batch["reward_model"]["ground_truth"]
    2. non_tensor_batch["ground_truth"]
    3. Falls back to decoding the final answer span from successful siblings
    """
    batch_size = len(batch)
    answers = [""] * batch_size

    for i in range(batch_size):
        # Try reward_model.ground_truth
        rm_data = batch.non_tensor_batch.get("reward_model")
        if rm_data is not None and i < len(rm_data):
            item = rm_data[i]
            if isinstance(item, dict):
                gt = item.get("ground_truth", "")
                if gt:
                    answers[i] = str(gt) if not isinstance(gt, str) else gt
                    continue

        # Try direct ground_truth field
        gt_field = batch.non_tensor_batch.get("ground_truth")
        if gt_field is not None and i < len(gt_field):
            gt = gt_field[i]
            if gt:
                answers[i] = str(gt) if not isinstance(gt, str) else gt

    return answers


# ---------------------------------------------------------------------------
# Student IG Computation (Structural / Heuristic)
# ---------------------------------------------------------------------------


def compute_student_ig_structural(
    batch: DataProto,
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute structural (heuristic) IG for student rollout.

    This is the v0.1 proxy that doesn't require model forward passes.
    It assigns IG=1.0 to turns that have retrieved documents (indicating
    the search was executed and returned content) and IG=0.0 otherwise.

    Args:
        batch: DataProto with response tokens
        tokenizer: tokenizer for decoding

    Returns:
        ig_student: (batch_size, response_len) IG values at turn-end positions
        query_mask: (batch_size, response_len) query span mask
        metrics: diagnostic metrics
    """
    response_ids = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    batch_size, response_len = response_ids.shape
    device = response_ids.device

    ig_student = torch.zeros(batch_size, response_len, dtype=torch.float32, device=device)
    query_mask = torch.zeros(batch_size, response_len, dtype=torch.bool, device=device)

    total_turns = 0
    turns_with_docs = 0
    total_docs = 0

    for i in range(batch_size):
        traj = parse_trajectory_tokens(
            response_ids=response_ids[i],
            response_mask=response_mask[i],
            tokenizer=tokenizer,
            sample_index=i,
        )

        for turn in traj.search_turns:
            total_turns += 1
            query_mask[i, turn.query_start:turn.query_end] = True

            if turn.documents:
                turns_with_docs += 1
                total_docs += len(turn.documents)
                # Place IG=1.0 at the end of the query span (turn boundary)
                end_pos = min(turn.query_end - 1, response_len - 1)
                if end_pos >= 0:
                    ig_student[i, end_pos] = 1.0

    metrics = {
        "ig_student_struct/total_turns": float(total_turns),
        "ig_student_struct/turns_with_docs": float(turns_with_docs),
        "ig_student_struct/turns_with_docs_frac": float(turns_with_docs / max(total_turns, 1)),
        "ig_student_struct/total_docs": float(total_docs),
        "ig_student_struct/mean_docs_per_turn": float(total_docs / max(turns_with_docs, 1)),
    }

    return ig_student, query_mask, metrics


# ---------------------------------------------------------------------------
# Teacher IG Computation (with Retrieval)
# ---------------------------------------------------------------------------


def compute_teacher_ig_structural(
    teacher_batch: DataProto,
    tokenizer: Any,
    retrieval_url: str = "",
    topk: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute structural IG for teacher rollout, optionally using retrieval.

    If retrieval_url is provided, executes teacher queries to get actual
    documents and assigns IG=1.0 where docs are found.
    Without retrieval_url, falls back to parsing the teacher response
    for embedded documents (which may not exist in hindsight mode).

    Args:
        teacher_batch: DataProto with teacher response tokens
        tokenizer: tokenizer for decoding
        retrieval_url: URL of retrieval service (optional)
        topk: number of docs to retrieve per query

    Returns:
        ig_teacher: (batch_size, response_len) IG values at turn-end positions
        query_mask: (batch_size, response_len) query span mask
        metrics: diagnostic metrics
    """
    response_ids = teacher_batch.batch["responses"]
    response_mask = teacher_batch.batch["response_mask"]
    batch_size, response_len = response_ids.shape
    device = response_ids.device

    ig_teacher = torch.zeros(batch_size, response_len, dtype=torch.float32, device=device)
    query_mask = torch.zeros(batch_size, response_len, dtype=torch.bool, device=device)

    total_turns = 0
    turns_with_docs = 0
    queries_to_retrieve: list[str] = []
    query_positions: list[tuple[int, int]] = []  # (sample_idx, end_pos)

    # Step 1: Parse teacher trajectories
    for i in range(batch_size):
        traj = parse_trajectory_tokens(
            response_ids=response_ids[i],
            response_mask=response_mask[i],
            tokenizer=tokenizer,
            sample_index=i,
        )

        for turn in traj.search_turns:
            total_turns += 1
            query_mask[i, turn.query_start:turn.query_end] = True
            end_pos = min(turn.query_end - 1, response_len - 1)

            if turn.documents:
                # Teacher already has docs embedded (e.g., from multi-turn rollout)
                turns_with_docs += 1
                if end_pos >= 0:
                    ig_teacher[i, end_pos] = 1.0
            elif turn.query_text and retrieval_url:
                # Need to retrieve docs for this teacher query
                queries_to_retrieve.append(turn.query_text)
                query_positions.append((i, end_pos))

    # Step 2: Execute retrieval for queries without embedded docs
    if queries_to_retrieve and retrieval_url:
        try:
            from verl.trainer.ppo.igsd_teacher_ig import retrieve_sync

            doc_results = retrieve_sync(queries_to_retrieve, retrieval_url, topk=topk)
            for (sample_idx, end_pos), docs in zip(query_positions, doc_results, strict=False):
                if docs:
                    turns_with_docs += 1
                    if end_pos >= 0:
                        ig_teacher[sample_idx, end_pos] = 1.0
        except Exception as e:
            logger.warning(f"Teacher retrieval failed: {e}")

    metrics = {
        "ig_teacher_struct/total_turns": float(total_turns),
        "ig_teacher_struct/turns_with_docs": float(turns_with_docs),
        "ig_teacher_struct/turns_with_docs_frac": float(turns_with_docs / max(total_turns, 1)),
        "ig_teacher_struct/queries_retrieved": float(len(queries_to_retrieve)),
    }

    return ig_teacher, query_mask, metrics


# ---------------------------------------------------------------------------
# Main Integration Entry Points (called from ray_trainer.py)
# ---------------------------------------------------------------------------


def inject_ig_student_into_batch(
    batch: DataProto,
    tokenizer: Any,
    config: Any,
    actor_rollout_wg: Any = None,
    log_prob_micro_batch_size_per_gpu: Optional[int] = None,
) -> dict[str, float]:
    """Compute and inject ig_student tensor into the batch.

    Called from the training loop BEFORE prepare_igsd_batch.

    Supports two modes:
    - Structural (heuristic): igsd_enable_ig_compute=True, igsd_enable_ig_forward=False
    - Model-based forward: igsd_enable_ig_compute=True, igsd_enable_ig_forward=True
      (requires actor_rollout_wg to be passed)

    Args:
        batch: DataProto that will be modified in-place
        tokenizer: tokenizer
        config: algorithm config
        actor_rollout_wg: (optional) actor worker group for model-based IG
        log_prob_micro_batch_size_per_gpu: (optional) per-GPU micro batch size for
            the actor's compute_log_prob path, used to pad pseudo batch correctly.

    Returns:
        metrics: diagnostic metrics
    """
    from verl.trainer.ppo.igsd_utils import _cfg_get

    if not bool(_cfg_get(config, "igsd_enable_ig_compute", False)):
        return {}

    use_forward = bool(_cfg_get(config, "igsd_enable_ig_forward", False))

    if use_forward and actor_rollout_wg is not None:
        # Model-based IG via forward pass
        from verl.trainer.ppo.igsd_ig_forward import compute_ig_forward

        answer_texts = _get_answer_texts(batch, tokenizer)
        num_cf = int(_cfg_get(config, "igsd_num_counterfactual", 3) or 3)
        max_ans_tok = int(_cfg_get(config, "igsd_max_answer_tokens", 128) or 128)

        ig_student, query_mask, metrics = compute_ig_forward(
            batch=batch,
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            answer_texts=answer_texts,
            num_counterfactual=num_cf,
            max_answer_tokens=max_ans_tok,
            log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        )
    else:
        # Structural heuristic (no model needed)
        ig_student, query_mask, metrics = compute_student_ig_structural(
            batch=batch,
            tokenizer=tokenizer,
        )

    # Inject into batch
    batch.batch["igsd_ig_student"] = ig_student
    # Also inject query mask for distillation targeting
    batch.batch["igsd_query_distill_mask"] = query_mask

    return metrics


def inject_ig_teacher_into_batch(
    teacher_batch: DataProto,
    tokenizer: Any,
    config: Any,
) -> dict[str, float]:
    """Compute and inject ig_teacher tensor into the teacher aux batch.

    Called from the training loop AFTER teacher rollout, BEFORE _attach_igsd_teacher_aux.

    Args:
        teacher_batch: DataProto for teacher aux (modified in-place)
        tokenizer: tokenizer
        config: algorithm config

    Returns:
        metrics: diagnostic metrics
    """
    from verl.trainer.ppo.igsd_utils import _cfg_get

    if not bool(_cfg_get(config, "igsd_enable_ig_compute", False)):
        return {}

    retrieval_url = str(_cfg_get(config, "igsd_retrieval_url", "") or "")
    topk = int(_cfg_get(config, "igsd_retrieval_topk", 3) or 3)

    ig_teacher, query_mask, metrics = compute_teacher_ig_structural(
        teacher_batch=teacher_batch,
        tokenizer=tokenizer,
        retrieval_url=retrieval_url,
        topk=topk,
    )

    # Inject into teacher batch
    teacher_batch.batch["igsd_ig_teacher"] = ig_teacher
    # Update query mask for teacher distillation targeting
    teacher_batch.batch["igsd_query_distill_mask"] = query_mask

    return metrics
