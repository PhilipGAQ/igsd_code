# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Model-based IG computation via actor forward pass.

This module implements the **counterfactual** IG computation using the actor model:

  IG_t = mean_k log π(a*_k | C_real,t) - (1/N) Σ_j mean_k log π(a*_k | C_rand,j,t)

where:
  - a* is the ground-truth answer tokens
  - C_real,t is the context including prompt + trajectory up to turn t with REAL docs
  - C_rand,j,t is the same context but with RANDOM docs replacing real ones at turn t

Key architectural constraint:
  The driver process does NOT hold the model weights. All forward passes go through
  `actor_rollout_wg.compute_log_prob()`, which dispatches to Ray workers.

Strategy (inspired by IGPO):
  1. For each sample × turn, construct "pseudo sequences" = [context | answer_tokens]
     - 1 sequence with real docs (factual)
     - N sequences with random docs (counterfactual)
  2. Batch ALL sequences across all samples/turns into ONE DataProto
  3. Make a SINGLE call to actor_rollout_wg.compute_log_prob()
  4. From returned log_probs, extract mean logprob of the answer span
  5. Compute IG per turn: real_logprob - mean(counterfactual_logprobs)

This replaces the structural heuristic with actual model-based probability estimation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.igsd_ig_compute import (
    ParsedTrajectory,
    build_random_doc_pool_from_batch,
    parse_trajectory_tokens,
)
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Core: Build pseudo sequences for counterfactual IG
# ---------------------------------------------------------------------------


def _build_counterfactual_tool_response(random_docs: list[str]) -> str:
    """Format random documents into the tool response format.

    Mimics the JSON format produced by the search tool.
    """
    import json

    parts = []
    for i, doc in enumerate(random_docs, 1):
        parts.append(f"Doc {i} (Title: Random Document {i})\n{doc}")
    result_text = "\n\n".join(parts)
    return json.dumps({"result": result_text})


def _sample_random_docs(doc_pool: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """Sample n random documents from the pool."""
    if len(doc_pool) <= n:
        return doc_pool[:]
    indices = rng.choice(len(doc_pool), size=n, replace=False)
    return [doc_pool[int(i)] for i in indices]


def compute_ig_forward(
    batch: DataProto,
    tokenizer: Any,
    actor_rollout_wg: Any,
    answer_texts: list[str],
    num_counterfactual: int = 3,
    max_answer_tokens: int = 128,
    log_prob_micro_batch_size_per_gpu: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute model-based counterfactual IG for the student batch.

    This function:
    1. Parses trajectories to find search turns
    2. Builds a doc pool from the batch for counterfactual sampling
    3. Constructs pseudo sequences [context | GT_answer] for real + counterfactual
    4. Calls actor_rollout_wg.compute_log_prob() ONCE for all sequences
    5. Extracts per-turn IG values

    Args:
        batch: DataProto with fields:
            - input_ids: (batch_size, seq_len) - full [prompt | response] tokens
            - attention_mask: (batch_size, seq_len)
            - position_ids: (batch_size, seq_len)
            - responses: (batch_size, response_len)
            - response_mask: (batch_size, response_len)
        tokenizer: tokenizer for encoding/decoding
        actor_rollout_wg: Ray worker group with compute_log_prob method
        answer_texts: ground-truth answer text per sample
        num_counterfactual: N - number of random doc replacements
        max_answer_tokens: max tokens for the answer span

    Returns:
        ig_tensor: (batch_size, response_len) with IG at turn-end positions
        query_mask: (batch_size, response_len) bool mask for query spans
        metrics: diagnostic metrics
    """
    response_ids = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]

    batch_size, response_len = response_ids.shape
    _, seq_len = input_ids.shape
    prompt_len = seq_len - response_len

    device = response_ids.device
    ig_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32, device=device)
    query_mask = torch.zeros(batch_size, response_len, dtype=torch.bool, device=device)

    rng = np.random.default_rng(42)
    t0 = time.perf_counter()

    # Step 1: Parse all trajectories
    all_trajectories: list[ParsedTrajectory] = []
    for i in range(batch_size):
        traj = parse_trajectory_tokens(
            response_ids=response_ids[i],
            response_mask=response_mask[i],
            tokenizer=tokenizer,
            sample_index=i,
        )
        all_trajectories.append(traj)

    # Step 2: Build random document pool from batch
    all_search_turns = [t.search_turns for t in all_trajectories]
    doc_pool = build_random_doc_pool_from_batch(all_search_turns)

    if not doc_pool:
        # No documents in batch at all; fall back to zero IG
        logger.warning("No documents found in batch; IG forward pass skipped.")
        return ig_tensor, query_mask, {"ig_forward/skipped_no_docs": 1.0}

    # Step 3: Build pseudo sequences
    # Each pseudo sequence is: [prompt_tokens + response_up_to_context_end | answer_tokens]
    # We need to track which pseudo sequence belongs to which (sample, turn, type)
    pseudo_prompts: list[list[int]] = []  # The "prompt" part (context)
    pseudo_answers: list[list[int]] = []  # The "response" part (GT answer)
    sequence_map: list[tuple[int, int, str]] = []  # (sample_idx, turn_idx, "real"|"cf_j")

    total_turns_with_docs = 0
    total_turns = 0

    for sample_idx, traj in enumerate(all_trajectories):
        answer_text = answer_texts[sample_idx] if sample_idx < len(answer_texts) else ""
        if not answer_text:
            # Mark query spans but skip IG computation
            for turn in traj.search_turns:
                query_mask[sample_idx, turn.query_start:turn.query_end] = True
            continue

        # Tokenize the answer (the GT we condition on)
        answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)[:max_answer_tokens]
        if not answer_token_ids:
            for turn in traj.search_turns:
                query_mask[sample_idx, turn.query_start:turn.query_end] = True
            continue

        # Get the full prompt (non-response) token ids for this sample
        # input_ids = [prompt_tokens | response_tokens]
        prompt_token_ids = input_ids[sample_idx, :prompt_len].tolist()
        # Remove left-padding from prompt (pad tokens at the start)
        pad_token_id = tokenizer.pad_token_id
        prompt_start = 0
        while prompt_start < len(prompt_token_ids) and prompt_token_ids[prompt_start] == pad_token_id:
            prompt_start += 1
        prompt_token_ids_clean = prompt_token_ids[prompt_start:]

        for turn_idx, turn in enumerate(traj.search_turns):
            total_turns += 1
            query_mask[sample_idx, turn.query_start:turn.query_end] = True

            if not turn.documents:
                # No docs at this turn, IG = 0 by definition
                continue

            total_turns_with_docs += 1

            # Build context = prompt + response[:tool_response_end]
            # tool_response_end is the position in the RESPONSE tensor (0-indexed)
            context_response_part = response_ids[sample_idx, :turn.tool_response_end].tolist()
            context_ids = prompt_token_ids_clean + context_response_part

            # === Real sequence (with actual docs) ===
            pseudo_prompts.append(context_ids)
            pseudo_answers.append(answer_token_ids)
            sequence_map.append((sample_idx, turn_idx, "real"))

            # === Counterfactual sequences (with random docs) ===
            # Build context WITHOUT the tool response, then append random docs
            context_before_tool_resp = response_ids[sample_idx, :turn.tool_response_start].tolist()
            context_prefix = prompt_token_ids_clean + context_before_tool_resp

            for cf_idx in range(num_counterfactual):
                n_docs = max(len(turn.documents), 1)
                random_docs = _sample_random_docs(doc_pool, n_docs, rng)
                cf_tool_resp_text = _build_counterfactual_tool_response(random_docs)
                cf_tool_resp_ids = tokenizer.encode(cf_tool_resp_text, add_special_tokens=False)

                cf_context_ids = context_prefix + cf_tool_resp_ids
                pseudo_prompts.append(cf_context_ids)
                pseudo_answers.append(answer_token_ids)
                sequence_map.append((sample_idx, turn_idx, f"cf_{cf_idx}"))

    if not pseudo_prompts:
        # No turns with documents or no valid answers
        metrics = {
            "ig_forward/total_turns": float(total_turns),
            "ig_forward/turns_with_docs": 0.0,
            "ig_forward/pseudo_sequences": 0.0,
        }
        return ig_tensor, query_mask, metrics

    # Step 4: Pack into DataProto and call compute_log_prob
    # Chunk if too many sequences (to avoid OOM on workers)
    MAX_PSEUDO_BATCH = 256  # Max sequences per forward call
    if len(pseudo_prompts) <= MAX_PSEUDO_BATCH:
        log_probs_all, answer_len = _call_compute_log_prob(
            pseudo_prompts=pseudo_prompts,
            pseudo_answers=pseudo_answers,
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        )
    else:
        # Process in chunks
        log_probs_all = []
        answer_len = 0
        for chunk_start in range(0, len(pseudo_prompts), MAX_PSEUDO_BATCH):
            chunk_end = min(chunk_start + MAX_PSEUDO_BATCH, len(pseudo_prompts))
            chunk_lps, chunk_ans_len = _call_compute_log_prob(
                pseudo_prompts=pseudo_prompts[chunk_start:chunk_end],
                pseudo_answers=pseudo_answers[chunk_start:chunk_end],
                tokenizer=tokenizer,
                actor_rollout_wg=actor_rollout_wg,
                log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
            )
            log_probs_all.extend(chunk_lps)
            answer_len = max(answer_len, chunk_ans_len)
        logger.info(
            f"IG forward: processed {len(pseudo_prompts)} sequences in "
            f"{(len(pseudo_prompts) + MAX_PSEUDO_BATCH - 1) // MAX_PSEUDO_BATCH} chunks"
        )

    t_forward_done = time.perf_counter()

    # Step 5: Compute per-turn IG from returned log_probs
    # log_probs_all[i] = mean log-prob of answer tokens for pseudo sequence i
    ig_values_all: list[float] = []
    real_lp_all: list[float] = []
    cf_lp_all: list[float] = []
    turn_results: dict[tuple[int, int], dict] = {}  # (sample_idx, turn_idx) -> {"real": val, "cf": [vals]}

    for seq_idx, (sample_idx, turn_idx, seq_type) in enumerate(sequence_map):
        key = (sample_idx, turn_idx)
        if key not in turn_results:
            turn_results[key] = {"real": None, "cf": []}
        if seq_type == "real":
            turn_results[key]["real"] = log_probs_all[seq_idx]
        else:
            turn_results[key]["cf"].append(log_probs_all[seq_idx])

    for (sample_idx, turn_idx), result in turn_results.items():
        if result["real"] is None or not result["cf"]:
            continue

        real_lp = result["real"]
        mean_cf_lp = sum(result["cf"]) / len(result["cf"])
        ig = real_lp - mean_cf_lp

        real_lp_all.append(real_lp)
        cf_lp_all.append(mean_cf_lp)

        # Clamp for numerical stability
        ig = max(min(ig, 10.0), -10.0)

        # Place IG at the turn-end position in the response tensor
        turn = all_trajectories[sample_idx].search_turns[turn_idx]
        end_pos = min(turn.query_end - 1, response_len - 1)
        if end_pos >= 0:
            ig_tensor[sample_idx, end_pos] = ig

        ig_values_all.append(ig)

    t_end = time.perf_counter()

    # Sequence length stats (for OOM monitoring)
    prompt_lens = [len(p) for p in pseudo_prompts]
    answer_lens = [len(a) for a in pseudo_answers]
    total_lens = [pl + al for pl, al in zip(prompt_lens, answer_lens)]

    # Answer coverage: how many samples had valid answers
    samples_with_answer = sum(1 for a in answer_texts if a)

    # Metrics
    metrics = {
        # Turn statistics
        "ig_forward/total_turns": float(total_turns),
        "ig_forward/turns_with_docs": float(total_turns_with_docs),
        "ig_forward/pseudo_sequences": float(len(pseudo_prompts)),
        "ig_forward/doc_pool_size": float(len(doc_pool)),
        # IG distribution
        "ig_forward/mean_ig": float(np.mean(ig_values_all)) if ig_values_all else 0.0,
        "ig_forward/positive_ig_frac": float(
            sum(1 for v in ig_values_all if v > 0) / max(len(ig_values_all), 1)
        ),
        # Raw logprob diagnostics (detect degenerate distributions)
        "ig_forward/real_logprob_mean": float(np.mean(real_lp_all)) if real_lp_all else 0.0,
        "ig_forward/cf_logprob_mean": float(np.mean(cf_lp_all)) if cf_lp_all else 0.0,
        # Sequence length stats (OOM risk monitoring)
        "ig_forward/max_seq_len": float(max(total_lens)) if total_lens else 0.0,
        "ig_forward/mean_seq_len": float(np.mean(total_lens)) if total_lens else 0.0,
        "ig_forward/mean_prompt_len": float(np.mean(prompt_lens)) if prompt_lens else 0.0,
        "ig_forward/mean_answer_len": float(np.mean(answer_lens)) if answer_lens else 0.0,
        # Coverage
        "ig_forward/samples_with_answer": float(samples_with_answer),
        "ig_forward/samples_total": float(batch_size),
        # Timing (seconds)
        "ig_forward/time_total_sec": t_end - t0,
        "ig_forward/time_forward_sec": t_forward_done - t0,
        # Chunking
        "ig_forward/num_chunks": float(max(1, (len(pseudo_prompts) + MAX_PSEUDO_BATCH - 1) // MAX_PSEUDO_BATCH)),
    }
    if ig_values_all:
        metrics["ig_forward/ig_std"] = float(np.std(ig_values_all))
        metrics["ig_forward/ig_max"] = float(max(ig_values_all))
        metrics["ig_forward/ig_min"] = float(min(ig_values_all))
    if real_lp_all:
        metrics["ig_forward/real_logprob_std"] = float(np.std(real_lp_all))
        metrics["ig_forward/cf_logprob_std"] = float(np.std(cf_lp_all))

    return ig_tensor, query_mask, metrics


# ---------------------------------------------------------------------------
# Helper: Pack pseudo sequences and call compute_log_prob
# ---------------------------------------------------------------------------


def _call_compute_log_prob(
    pseudo_prompts: list[list[int]],
    pseudo_answers: list[list[int]],
    tokenizer: Any,
    actor_rollout_wg: Any,
    log_prob_micro_batch_size_per_gpu: int | None = None,
    *,
    sanitize_nonfinite: bool = True,
) -> tuple[list[float], int]:
    """Pack pseudo sequences into DataProto and call compute_log_prob.

    Following verl's convention:
    - input_ids = [prompt | response]  (left-padded prompts, right-padded responses)
    - attention_mask = 1 where not padding
    - position_ids = cumulative positions for non-pad tokens
    - prompts = prompt portion
    - responses = response portion (the answer tokens)
    - response_mask = 1 for valid answer tokens, 0 for padding

    The worker's compute_log_prob returns log_probs at the response positions.
    We extract mean(log_probs[0:actual_answer_len]) for each sequence.

    Args:
        pseudo_prompts: list of token id lists (the "context" for each sequence)
        pseudo_answers: list of token id lists (the "GT answer" for each sequence)
        tokenizer: tokenizer (for pad_token_id, eos_token_id)
        actor_rollout_wg: worker group with compute_log_prob
        log_prob_micro_batch_size_per_gpu: per-GPU micro batch size used by the
            worker for compute_log_prob. The total batch must be divisible by
            world_size * micro_batch_size_per_gpu to avoid assertion errors.
        sanitize_nonfinite: preserve the legacy behavior of replacing non-finite
            sequence scores with zero. Callers that need fail-closed validity
            tracking can disable sanitization and filter the returned values.

    Returns:
        mean_log_probs: list of mean log-prob of answer tokens per sequence
        answer_len: padded answer length (all answers are padded to same length)
    """
    num_sequences = len(pseudo_prompts)
    pad_token_id = tokenizer.pad_token_id

    # Determine max lengths for padding
    max_prompt_len = max(len(p) for p in pseudo_prompts)
    # All answers should be same length (same GT per sample), but we pad to max just in case
    max_answer_len = max(len(a) for a in pseudo_answers)

    # Build tensors with LEFT-padded prompts and RIGHT-padded responses
    # This matches verl's expected format: [left_pad ... prompt_tokens | response_tokens ... right_pad]
    all_prompt_ids = []
    all_response_ids = []
    actual_answer_lens = []

    for i in range(num_sequences):
        # Left-pad prompt
        prompt = pseudo_prompts[i]
        prompt_pad_len = max_prompt_len - len(prompt)
        padded_prompt = [pad_token_id] * prompt_pad_len + prompt
        all_prompt_ids.append(padded_prompt)

        # Right-pad response (answer)
        answer = pseudo_answers[i]
        actual_answer_lens.append(len(answer))
        answer_pad_len = max_answer_len - len(answer)
        padded_answer = answer + [pad_token_id] * answer_pad_len
        all_response_ids.append(padded_answer)

    prompt_tensor = torch.tensor(all_prompt_ids, dtype=torch.long)   # (N, max_prompt_len)
    response_tensor = torch.tensor(all_response_ids, dtype=torch.long)  # (N, max_answer_len)

    # input_ids = cat(prompt, response)
    input_ids = torch.cat([prompt_tensor, response_tensor], dim=1)  # (N, max_prompt_len + max_answer_len)

    # attention_mask: 1 for non-pad tokens
    attention_mask = (input_ids != pad_token_id).long()

    # response_mask: 1 for valid answer tokens, 0 for padding in the response portion
    # (our pseudo responses are pure GT answer tokens + right-padding)
    response_mask = (response_tensor != pad_token_id).long()

    # position_ids: cumulative positions for non-pad tokens
    # For left-padded input: position_ids[i] = cumsum(attention_mask[i]) - 1, clamped to 0
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids = position_ids.clamp(min=0)
    # Zero out padding positions
    position_ids = position_ids * attention_mask

    # Build DataProto
    batch_td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_tensor,
            "response_mask": response_mask,
            "prompts": prompt_tensor,
        },
        batch_size=num_sequences,
    )
    pseudo_batch = DataProto.from_tensordict(batch_td)

    # Pad to divisor: world_size * micro_batch_size_per_gpu
    # The worker splits data evenly across world_size ranks, then each rank chunks
    # its portion into micro-batches of size micro_batch_size_per_gpu. So the total
    # batch must be divisible by (world_size * micro_batch_size_per_gpu).
    world_size = max(int(getattr(actor_rollout_wg, "world_size", 1)), 1)
    mbs = max(int(log_prob_micro_batch_size_per_gpu or 1), 1)
    size_divisor = world_size * mbs
    pseudo_batch_padded, pad_size = pad_dataproto_to_divisor(pseudo_batch, size_divisor)

    # Convert to no-padding format and call compute_log_prob
    batch_td_padded = pseudo_batch_padded.to_tensordict()
    batch_td_padded = left_right_2_no_padding(batch_td_padded)

    # Add metadata expected by the worker
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(
        batch_td_padded,
        calculate_entropy=False,
        compute_loss=False,
        temperature=1.0,  # Use temperature=1.0 for IG (raw logprobs)
    )

    # Call forward pass on workers
    output = actor_rollout_wg.compute_log_prob(batch_td_padded)

    # Extract log_probs and convert back to padded format
    log_probs = tu.get(output, "log_probs")
    log_probs = no_padding_2_padding(log_probs, batch_td_padded)

    # Unpad if we added padding
    log_probs_full = log_probs  # (N + pad_size, max_answer_len)
    # Remove padding samples
    log_probs_valid = log_probs_full[:num_sequences]  # (N, max_answer_len)

    # Compute mean log-prob for each sequence over valid answer tokens
    mean_log_probs: list[float] = []
    for i in range(num_sequences):
        actual_len = actual_answer_lens[i]
        if actual_len == 0:
            mean_log_probs.append(0.0)
            continue
        # log_probs are at response positions; take mean over actual answer tokens
        token_lps = log_probs_valid[i, :actual_len]
        mean_lp = token_lps.mean().item()
        # Keep the legacy IG-forward fallback by default. G1 disables it so a
        # numerical failure cannot look like an unusually good log-probability.
        if sanitize_nonfinite and not np.isfinite(mean_lp):
            mean_lp = 0.0
        mean_log_probs.append(mean_lp)

    return mean_log_probs, max_answer_len
