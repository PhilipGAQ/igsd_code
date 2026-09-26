# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Teacher IG computation for IGSD gate activation.

This module provides the logic to:
1. After teacher rollout, parse the teacher's response to find search turns
2. Execute teacher queries against the retrieval service to get teacher documents
3. Compute IG_teacher using the same counterfactual method
4. Compare IG_teacher vs IG_student to produce gate values

The gate decides whether teacher knowledge at each turn is worth distilling:
  gate_t = σ(β * (IG_teacher,t - IG_student,t))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import aiohttp
import numpy as np
import torch

from verl.trainer.ppo.igsd_ig_compute import (
    ParsedTrajectory,
    SearchTurn,
    _answer_logprob,
    _format_counterfactual_tool_response,
    _sample_random_docs,
    build_query_span_mask,
    parse_trajectory_tokens,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Retrieval Client (for executing teacher queries at driver level)
# ---------------------------------------------------------------------------


async def _retrieve_async(
    queries: list[str],
    retrieval_url: str,
    topk: int = 3,
    timeout: float = 30.0,
) -> list[list[str]]:
    """Execute search queries against the retrieval service.

    Args:
        queries: list of query strings
        retrieval_url: URL of the retrieval service
        topk: number of documents to retrieve per query
        timeout: request timeout in seconds

    Returns:
        list of document lists, one per query
    """
    if not queries:
        return []

    # Match SearchTool's request schema. In particular, return_scores=True wraps
    # each corpus row under ``document``; parsing that shape here keeps teacher
    # IG evidence text consistent with normal tool rollouts.
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                retrieval_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Retrieval service returned status {resp.status}")
                    return [[] for _ in queries]
                data = await resp.json()
                results = data.get("result", [])
                # results is list of list of documents (or list of list of dicts)
                doc_lists = []
                for per_query_result in results:
                    docs = []
                    if isinstance(per_query_result, list):
                        for doc_idx, item in enumerate(per_query_result):
                            if isinstance(item, dict):
                                doc_text = item.get("document", item)
                                if isinstance(doc_text, dict):
                                    contents = str(doc_text.get("contents", ""))
                                    if contents:
                                        lines = contents.split("\n")
                                        title = lines[0] if lines else "No Title"
                                        text = "\n".join(lines[1:]) if len(lines) > 1 else contents
                                    else:
                                        title = str(doc_text.get("title", "No Title"))
                                        text = str(doc_text.get("text", ""))
                                    docs.append(f"Doc {doc_idx + 1} (Title: {title})\n{text}")
                                else:
                                    docs.append(str(doc_text))
                            else:
                                docs.append(str(item))
                    doc_lists.append(docs)
                if len(doc_lists) < len(queries):
                    doc_lists.extend([[] for _ in range(len(queries) - len(doc_lists))])
                return doc_lists[: len(queries)]
    except Exception as e:
        logger.warning(f"Retrieval request failed: {e}")
        return [[] for _ in queries]


def retrieve_sync(
    queries: list[str],
    retrieval_url: str,
    topk: int = 3,
    timeout: float = 30.0,
) -> list[list[str]]:
    """Synchronous wrapper for retrieval."""
    if not queries:
        return []
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_retrieve_async(queries, retrieval_url, topk, timeout))
        else:
            # The trainer driver may already be inside an event loop. Run the
            # blocking sync wrapper in a separate thread so nested asyncio usage
            # does not fail.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    _retrieve_async(queries, retrieval_url, topk, timeout),
                )
                return future.result(timeout=timeout + 5)
    except Exception as e:
        logger.warning(f"Sync retrieval failed: {e}")
        return [[] for _ in queries]


# ---------------------------------------------------------------------------
# Teacher IG Computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_teacher_ig_for_batch(
    model: Any,
    tokenizer: Any,
    teacher_response_ids: torch.Tensor,
    teacher_response_mask: torch.Tensor,
    answer_texts: list[str],
    random_doc_pool: list[str],
    retrieval_url: str,
    topk: int = 3,
    num_counterfactual: int = 3,
    max_answer_tokens: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[list[SearchTurn]], dict[str, float]]:
    """Compute IG for teacher rollout responses.

    Unlike student IG (which uses documents already in the trajectory),
    teacher IG requires executing the teacher's queries against the retrieval
    service to get the actual documents.

    Args:
        model: policy model for answer log-prob computation
        tokenizer: tokenizer
        teacher_response_ids: (batch_size, response_len) teacher response tokens
        teacher_response_mask: (batch_size, response_len) mask
        answer_texts: ground-truth answers per sample
        random_doc_pool: documents for counterfactual (shared with student)
        retrieval_url: URL of the retrieval service
        topk: number of docs to retrieve
        num_counterfactual: N for counterfactual computation
        max_answer_tokens: max answer tokens
        device: compute device

    Returns:
        ig_teacher: (batch_size, response_len) IG values at turn-end positions
        query_mask: (batch_size, response_len) query span mask
        teacher_turns: parsed teacher search turns
        metrics: diagnostic metrics
    """
    batch_size, response_len = teacher_response_ids.shape
    if device is None:
        device = teacher_response_ids.device

    ig_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32, device=device)
    query_mask = torch.zeros(batch_size, response_len, dtype=torch.bool, device=device)

    # Step 1: Parse teacher trajectories
    teacher_turns_all: list[list[SearchTurn]] = []
    for i in range(batch_size):
        traj = parse_trajectory_tokens(
            response_ids=teacher_response_ids[i],
            response_mask=teacher_response_mask[i],
            tokenizer=tokenizer,
            sample_index=i,
        )
        teacher_turns_all.append(traj.search_turns)

    # Step 2: Collect all teacher queries for batch retrieval
    all_queries: list[str] = []
    query_mapping: list[tuple[int, int]] = []  # (sample_idx, turn_idx)
    for sample_idx, turns in enumerate(teacher_turns_all):
        for turn_idx, turn in enumerate(turns):
            if turn.query_text:
                all_queries.append(turn.query_text)
                query_mapping.append((sample_idx, turn_idx))

    # Step 3: Execute retrieval for all teacher queries
    if all_queries and retrieval_url:
        doc_results = retrieve_sync(all_queries, retrieval_url, topk=topk)
        # Assign retrieved docs to turns
        for (sample_idx, turn_idx), docs in zip(query_mapping, doc_results, strict=False):
            teacher_turns_all[sample_idx][turn_idx].documents = docs
    else:
        doc_results = []

    # Step 4: Compute IG for each turn
    total_turns = 0
    ig_values_all = []

    for sample_idx in range(batch_size):
        turns = teacher_turns_all[sample_idx]
        answer_text = answer_texts[sample_idx] if sample_idx < len(answer_texts) else ""
        if not turns or not answer_text:
            continue

        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)[:max_answer_tokens]
        if not answer_ids:
            continue

        for turn in turns:
            query_mask[sample_idx, turn.query_start:turn.query_end] = True

            if not turn.documents:
                continue

            # For teacher IG, we need to build a context that includes the teacher's
            # retrieved documents. Since the teacher response may not include the
            # tool response in the same format (it was generated fresh), we construct
            # the context synthetically.
            ig_value = _compute_teacher_turn_ig(
                model=model,
                tokenizer=tokenizer,
                response_ids=teacher_response_ids[sample_idx],
                turn=turn,
                answer_ids=answer_ids,
                random_doc_pool=random_doc_pool,
                num_counterfactual=num_counterfactual,
                device=device,
            )

            end_pos = min(turn.query_end - 1, response_len - 1)
            if end_pos >= 0:
                ig_tensor[sample_idx, end_pos] = ig_value

            total_turns += 1
            ig_values_all.append(ig_value)

    metrics = {
        "ig_teacher/total_turns": float(total_turns),
        "ig_teacher/mean_ig": float(sum(ig_values_all) / max(len(ig_values_all), 1)),
        "ig_teacher/queries_executed": float(len(all_queries)),
        "ig_teacher/docs_retrieved": float(sum(len(d) for d in doc_results)),
    }
    if ig_values_all:
        metrics["ig_teacher/ig_std"] = float(np.std(ig_values_all))
        metrics["ig_teacher/positive_ig_frac"] = float(
            sum(1 for v in ig_values_all if v > 0) / len(ig_values_all)
        )

    return ig_tensor, query_mask, teacher_turns_all, metrics


@torch.no_grad()
def _compute_teacher_turn_ig(
    model: Any,
    tokenizer: Any,
    response_ids: torch.Tensor,
    turn: SearchTurn,
    answer_ids: list[int],
    random_doc_pool: list[str],
    num_counterfactual: int,
    device: torch.device,
) -> float:
    """Compute IG for a single teacher turn.

    The teacher's context includes:
    - Everything before the tool_response_start (the teacher's thinking + query)
    - The retrieved documents (teacher's search results)

    IG = logprob(answer | context + real_docs) - mean(logprob(answer | context + random_docs))
    """
    # Build the context prefix: teacher response up to query end
    context_prefix_ids = response_ids[:turn.tool_response_start].tolist()

    # Build real doc context: prefix + formatted teacher docs
    real_docs_text = _format_counterfactual_tool_response(turn.documents)
    real_docs_ids = tokenizer.encode(real_docs_text, add_special_tokens=False)
    real_context_ids = context_prefix_ids + real_docs_ids

    real_logprob = _answer_logprob(model, tokenizer, real_context_ids, answer_ids, device)

    # Build counterfactual contexts
    counterfactual_logprobs = []
    for _ in range(num_counterfactual):
        n_docs = max(len(turn.documents), 1)
        random_docs = _sample_random_docs(random_doc_pool, n_docs)
        cf_docs_text = _format_counterfactual_tool_response(random_docs)
        cf_docs_ids = tokenizer.encode(cf_docs_text, add_special_tokens=False)
        cf_context_ids = context_prefix_ids + cf_docs_ids

        cf_logprob = _answer_logprob(model, tokenizer, cf_context_ids, answer_ids, device)
        counterfactual_logprobs.append(cf_logprob)

    if not counterfactual_logprobs:
        return 0.0

    mean_cf_logprob = sum(counterfactual_logprobs) / len(counterfactual_logprobs)
    return real_logprob - mean_cf_logprob


# ---------------------------------------------------------------------------
# Gate Computation (Combines Student and Teacher IG)
# ---------------------------------------------------------------------------


def compute_ig_gate(
    ig_student: torch.Tensor,
    ig_teacher: torch.Tensor,
    beta: float = 5.0,
    margin: float = 0.0,
    gate_mode: str = "sigmoid",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the IG gate from student and teacher IG values.

    gate_t = σ(β * (IG_teacher,t - IG_student,t - margin))

    Args:
        ig_student: (batch_size, response_len) student IG values
        ig_teacher: (batch_size, response_len) teacher IG values
        beta: temperature parameter for sigmoid gate
        margin: minimum IG advantage required to open gate
        gate_mode: "sigmoid", "hard", or "none"

    Returns:
        gate: (batch_size, response_len) gate values in [0, 1]
        metrics: diagnostic metrics
    """
    delta = ig_teacher - ig_student

    # Only compute at positions where either IG is non-zero
    valid_mask = (ig_student != 0) | (ig_teacher != 0)

    if gate_mode == "none":
        gate = torch.ones_like(delta)
    elif gate_mode == "hard":
        gate = (delta > margin).float()
    elif gate_mode == "sigmoid":
        gate = torch.sigmoid(beta * (delta - margin))
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")

    # Metrics
    valid_count = valid_mask.float().sum().item()
    metrics = {}
    if valid_count > 0:
        valid_delta = delta[valid_mask]
        valid_gate = gate[valid_mask]
        metrics["ig_gate/delta_mean"] = float(valid_delta.mean().item())
        metrics["ig_gate/delta_std"] = float(valid_delta.std().item()) if valid_count > 1 else 0.0
        metrics["ig_gate/gate_mean"] = float(valid_gate.mean().item())
        metrics["ig_gate/gate_open_frac"] = float((valid_gate > 0.5).float().mean().item())
        metrics["ig_gate/bad_teacher_frac"] = float((valid_delta <= 0).float().mean().item())
        metrics["ig_gate/valid_turn_count"] = float(valid_count)
    else:
        metrics["ig_gate/delta_mean"] = 0.0
        metrics["ig_gate/gate_mean"] = 0.0
        metrics["ig_gate/gate_open_frac"] = 0.0
        metrics["ig_gate/bad_teacher_frac"] = 0.0
        metrics["ig_gate/valid_turn_count"] = 0.0

    return gate, metrics
