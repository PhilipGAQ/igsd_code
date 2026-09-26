# Copyright 2026 The SearchAgent-Zero authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Utilities for environment-verified hindsight distillation.

The privileged teacher sees a failed prefix plus successful sibling queries, but
never the ground-truth answer or successful observations. The student is trained
on the original, unprivileged prefix; the auxiliary loss is weighted by an
environment-derived, paired evidence-utility contrast.
"""

from __future__ import annotations

import json
import math
import re
import string
import time
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.igsd_candidate import is_eligible_disagreement
from verl.trainer.ppo.igsd_ig_compute import TOOL_CALL_PATTERN, SearchTurn
from verl.trainer.ppo.igsd_ig_forward import _call_compute_log_prob
from verl.trainer.ppo.igsd_teacher_ig import retrieve_sync
from verl.trainer.ppo.igsd_utils import compute_gate, effective_gate_margin


def extract_answer_aliases(batch: DataProto, sample_idx: int, max_aliases: int = 3) -> list[str]:
    """Extract normalized answer aliases without stringifying container objects."""

    value: Any = None
    reward_model = batch.non_tensor_batch.get("reward_model")
    if reward_model is not None and sample_idx < len(reward_model):
        item = reward_model[sample_idx]
        if isinstance(item, dict):
            value = item.get("ground_truth")
    if value is None and "ground_truth" in batch.non_tensor_batch:
        value = batch.non_tensor_batch["ground_truth"][sample_idx]

    if isinstance(value, dict):
        value = value.get("target", value.get("answers", value.get("answer", [])))
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        candidates = list(value)
    elif value is None:
        candidates = []
    else:
        candidates = str(value).split("<|answer_split|>")

    aliases: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text not in aliases:
            aliases.append(text)
        if len(aliases) >= max(max_aliases, 1):
            break
    return aliases


def canonical_search_action(query: str) -> str:
    return canonical_search_action_queries([query])


def canonical_search_action_queries(queries: list[str] | tuple[str, ...]) -> str:
    """Serialize a complete ASearcher search call without flattening slots."""

    query_list = [str(query).strip() for query in queries if str(query).strip()]
    payload = {"name": "search", "arguments": {"query_list": query_list}}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def _canonical_search_action_ids_and_query_mask(query: str, tokenizer: Any) -> tuple[list[int], list[int]]:
    """Encode a canonical search action and mark only the serialized query value.

    The target sequence remains the full tool call so model inputs stay identical
    across span modes. In ``query_only`` mode the distillation loss is restricted
    to the JSON string content inside ``query_list``.
    """

    action_text = canonical_search_action(query)
    input_ids = tokenizer.encode(action_text, add_special_tokens=False)
    query_text = query.strip()
    if not input_ids or not query_text:
        return input_ids, [0] * len(input_ids)

    escaped_query = json.dumps(query_text, ensure_ascii=False)
    query_literal_start = action_text.find(escaped_query)
    if query_literal_start < 0 or len(escaped_query) < 2:
        return input_ids, [0] * len(input_ids)
    query_start = query_literal_start + 1
    query_end = query_literal_start + len(escaped_query) - 1

    mask = [0] * len(input_ids)
    try:
        encoded = tokenizer(action_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        offset_ids = encoded.get("input_ids")
        if offsets is not None and offset_ids == input_ids:
            for idx, (start, end) in enumerate(offsets):
                if end > query_start and start < query_end:
                    mask[idx] = 1
            return input_ids, mask
    except (NotImplementedError, TypeError, ValueError, AttributeError):
        pass

    # Fallback for non-fast tokenizers. This can be slightly conservative around
    # tokenizer merge boundaries, but keeps query-only from silently becoming
    # full-tool-call distillation.
    prefix_ids = tokenizer.encode(action_text[:query_start], add_special_tokens=False)
    prefix_plus_query_ids = tokenizer.encode(action_text[:query_end], add_special_tokens=False)
    start_idx = min(len(prefix_ids), len(input_ids))
    end_idx = min(max(len(prefix_plus_query_ids), start_idx), len(input_ids))
    for idx in range(start_idx, end_idx):
        mask[idx] = 1
    return input_ids, mask


def _canonical_search_action_queries_ids_and_query_mask(
    queries: list[str] | tuple[str, ...], tokenizer: Any
) -> tuple[list[int], list[int]]:
    """Encode a multi-query action and mask each query literal only."""

    action_text = canonical_search_action_queries(queries)
    input_ids = tokenizer.encode(action_text, add_special_tokens=False)
    if not input_ids:
        return input_ids, []
    mask = [0] * len(input_ids)
    # Character spans are unambiguous in the serialized JSON.  Fast tokenizer
    # offsets are preferred; the prefix re-encode fallback mirrors the legacy
    # single-query helper and remains conservative for slow tokenizers.
    raw_values = [str(query).strip() for query in queries if str(query).strip()]
    cursor = 0
    for query in raw_values:
        literal = json.dumps(query, ensure_ascii=False)
        literal_start = action_text.find(literal, cursor)
        if literal_start < 0:
            continue
        query_start = literal_start + 1
        query_end = literal_start + len(literal) - 1
        try:
            encoded = tokenizer(action_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping")
            offset_ids = encoded.get("input_ids")
            if offsets is not None and offset_ids is not None and list(offset_ids) == input_ids:
                for idx, (start, end) in enumerate(offsets):
                    if int(end) > query_start and int(start) < query_end:
                        mask[idx] = 1
            else:
                prefix_ids = tokenizer.encode(action_text[:query_start], add_special_tokens=False)
                prefix_plus_query_ids = tokenizer.encode(action_text[:query_end], add_special_tokens=False)
                start_idx = min(len(prefix_ids), len(input_ids))
                end_idx = min(max(len(prefix_plus_query_ids), start_idx), len(input_ids))
                for idx in range(start_idx, end_idx):
                    mask[idx] = 1
        except (NotImplementedError, TypeError, ValueError, AttributeError):
            prefix_ids = tokenizer.encode(action_text[:query_start], add_special_tokens=False)
            prefix_plus_query_ids = tokenizer.encode(action_text[:query_end], add_special_tokens=False)
            start_idx = min(len(prefix_ids), len(input_ids))
            end_idx = min(max(len(prefix_plus_query_ids), start_idx), len(input_ids))
            for idx in range(start_idx, end_idx):
                mask[idx] = 1
        cursor = literal_start + len(literal)
    return input_ids, mask


def _distill_action_ids_and_query_mask(
    example: dict[str, Any],
    tokenizer: Any,
    distill_target: str,
) -> tuple[list[int], list[int]]:
    """Return the shared teacher-forcing action and its query-value mask."""

    if distill_target == "teacher_action":
        queries = example.get("teacher_queries")
        if queries:
            return _canonical_search_action_queries_ids_and_query_mask(queries, tokenizer)
        return _canonical_search_action_ids_and_query_mask(example["teacher_query"], tokenizer)
    if distill_target != "student_on_policy":
        raise ValueError(
            f"Unsupported IGSD distill_target {distill_target!r}; "
            "expected 'teacher_action' or 'student_on_policy'"
        )

    response_ids = [int(token_id) for token_id in example.get("student_action_ids", [])]
    query_mask = [int(value) for value in example.get("student_action_query_mask", [])]
    if not response_ids:
        raise ValueError("student_on_policy distillation requires non-empty student_action_ids")
    if len(query_mask) != len(response_ids):
        raise ValueError(
            "student_on_policy query mask must align with the sampled student action: "
            f"{len(query_mask)=}, {len(response_ids)=}"
        )
    return response_ids, query_mask


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a finite Pearson correlation for one-dimensional diagnostics."""

    if left.numel() < 2 or right.numel() != left.numel():
        return 0.0
    left = left.float()
    right = right.float()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.square().sum().sqrt() * right_centered.square().sum().sqrt()
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= torch.finfo(torch.float32).eps:
        return 0.0
    correlation = (left_centered * right_centered).sum() / denominator
    return float(correlation.clamp(min=-1.0, max=1.0).item())


def apply_action_likelihood_gate(
    batch: DataProto,
    config: Any,
    global_step: int,
    lambda_coef: float,
) -> dict[str, float]:
    """Replace paired-IG weights with a detached action-likelihood gate.

    The target remains the teacher-generated canonical action. The privileged
    teacher and unprivileged old policy force-decode that same action, and the
    length-normalized log-probability gap over query-value tokens supplies the
    row-level reliability signal. Paired IG stays attached for matched-coverage
    diagnostics but does not affect the resulting distillation weights.
    """

    required = {
        "response_mask",
        "old_log_probs",
        "igsd_teacher_log_probs",
        "igsd_query_token_mask",
        "igsd_query_distill_mask",
    }
    missing = sorted(required - set(batch.batch.keys()))
    if missing:
        raise KeyError(f"Action-likelihood gating requires batch fields {missing}")
    if not math.isfinite(lambda_coef) or lambda_coef < 0.0:
        raise ValueError(f"IGSD lambda must be finite and non-negative, got {lambda_coef}")

    response_mask = batch.batch["response_mask"].bool()
    query_mask = batch.batch["igsd_query_token_mask"].bool() & response_mask
    distill_mask = batch.batch["igsd_query_distill_mask"].bool() & response_mask
    teacher_log_probs = batch.batch["igsd_teacher_log_probs"].detach().float()
    student_log_probs = batch.batch["old_log_probs"].detach().float()
    expected_shape = response_mask.shape
    for name, tensor in {
        "igsd_query_token_mask": query_mask,
        "igsd_query_distill_mask": distill_mask,
        "igsd_teacher_log_probs": teacher_log_probs,
        "old_log_probs": student_log_probs,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Action-likelihood field {name} must have shape {tuple(expected_shape)}, "
                f"got {tuple(tensor.shape)}"
            )

    finite = torch.isfinite(teacher_log_probs) & torch.isfinite(student_log_probs)
    query_token_count = query_mask.sum(dim=-1)
    valid_rows = query_token_count.gt(0) & ((~query_mask) | finite).all(dim=-1)
    safe_token_gap = torch.where(
        finite,
        teacher_log_probs - student_log_probs,
        torch.zeros_like(teacher_log_probs),
    )
    query_gap = (safe_token_gap * query_mask.float()).sum(dim=-1) / query_token_count.clamp_min(1).float()
    query_gap = torch.where(valid_rows, query_gap, torch.zeros_like(query_gap))

    gate = compute_gate(query_gap, config, global_step).detach().float()
    gate = torch.where(valid_rows, gate, torch.zeros_like(gate))
    expanded_gate = gate[:, None].expand_as(response_mask).clone()
    batch.batch["igsd_gate"] = expanded_gate
    batch.batch["igsd_turn_weight"] = expanded_gate.clone()
    batch.batch["igsd_distill_weights"] = distill_mask.float() * expanded_gate * float(lambda_coef)

    valid_gap = query_gap[valid_rows]
    valid_gate = gate[valid_rows]
    valid_query_count = query_token_count[valid_rows].float()
    teacher_query_mean = (
        torch.where(query_mask & finite, teacher_log_probs, torch.zeros_like(teacher_log_probs)).sum(dim=-1)
        / query_token_count.clamp_min(1).float()
    )[valid_rows]
    student_query_mean = (
        torch.where(query_mask & finite, student_log_probs, torch.zeros_like(student_log_probs)).sum(dim=-1)
        / query_token_count.clamp_min(1).float()
    )[valid_rows]
    full_span_token_count = response_mask.sum(dim=-1)
    full_span_valid = full_span_token_count.gt(0) & ((~response_mask) | finite).all(dim=-1)
    full_span_gap = (
        (safe_token_gap * response_mask.float()).sum(dim=-1)
        / full_span_token_count.clamp_min(1).float()
    )[valid_rows & full_span_valid]

    def scalar_mean(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else 0.0

    def scalar_std(values: torch.Tensor) -> float:
        return float(values.std(unbiased=False).item()) if values.numel() else 0.0

    def quantile(values: torch.Tensor, value: float) -> float:
        return float(torch.quantile(values, value).item()) if values.numel() else 0.0

    accepted = valid_gate.gt(0.0)
    gate_sum = valid_gate.sum()
    gate_square_sum = valid_gate.square().sum()
    gate_ess = (
        float((gate_sum.square() / gate_square_sum).item())
        if valid_gate.numel() and float(gate_square_sum.item()) > 0.0
        else 0.0
    )
    margin = effective_gate_margin(config, global_step)
    likelihood_positive = valid_gap.gt(margin)

    metrics = {
        "igsd/action_likelihood_valid_row_count": float(valid_rows.sum().item()),
        "igsd/action_likelihood_invalid_row_count": float((~valid_rows).sum().item()),
        "igsd/action_likelihood_valid_row_frac": float(valid_rows.float().mean().item()),
        "igsd/action_likelihood_gap_mean": scalar_mean(valid_gap),
        "igsd/action_likelihood_gap_std": scalar_std(valid_gap),
        "igsd/action_likelihood_gap_min": float(valid_gap.min().item()) if valid_gap.numel() else 0.0,
        "igsd/action_likelihood_gap_p10": quantile(valid_gap, 0.10),
        "igsd/action_likelihood_gap_p50": quantile(valid_gap, 0.50),
        "igsd/action_likelihood_gap_p90": quantile(valid_gap, 0.90),
        "igsd/action_likelihood_gap_max": float(valid_gap.max().item()) if valid_gap.numel() else 0.0,
        "igsd/action_likelihood_positive_frac": scalar_mean(likelihood_positive.float()),
        "igsd/action_likelihood_teacher_query_log_prob_mean": scalar_mean(teacher_query_mean),
        "igsd/action_likelihood_student_query_log_prob_mean": scalar_mean(student_query_mean),
        "igsd/action_likelihood_full_span_gap_mean": scalar_mean(full_span_gap),
        "igsd/action_likelihood_full_span_valid_row_count": float(
            (valid_rows & full_span_valid).sum().item()
        ),
        "igsd/action_likelihood_query_token_count_mean": scalar_mean(valid_query_count),
        "igsd/action_likelihood_gap_query_length_pearson": _pearson_correlation(
            valid_gap, valid_query_count
        ),
        "igsd/action_likelihood_gate_mean": scalar_mean(valid_gate),
        "igsd/action_likelihood_gate_accepted_frac": scalar_mean(accepted.float()),
        "igsd/action_likelihood_gate_accepted_mean": scalar_mean(valid_gate[accepted]),
        "igsd/action_likelihood_gate_effective_sample_size": gate_ess,
        "igsd/action_likelihood_gate_effective_sample_frac": gate_ess / max(valid_gate.numel(), 1),
        "igsd/m2_gate_mean": scalar_mean(gate),
        "igsd/m2_positive_delta_frac": scalar_mean(likelihood_positive.float()),
        "igsd/m2_gate_effective_sample_size": gate_ess,
        "igsd/m2_gate_effective_sample_frac": gate_ess / max(valid_gate.numel(), 1),
        "igsd/m2_active_distill_pair_count": float(accepted.sum().item()),
        "igsd/m2_active_distill_pair_frac": float(accepted.sum().item() / max(len(batch), 1)),
        "igsd/m2_routed_turn_count": float(accepted.sum().item()),
        "igsd/m2_routed_turn_frac": float(accepted.sum().item() / max(len(batch), 1)),
        "igsd/m2_routed_weight_mean": scalar_mean(gate),
        "igsd/m2_routed_weight_sum": float(gate.sum().item()),
        "igsd/m2_routed_weight_max": float(gate.max().item()) if gate.numel() else 0.0,
    }

    if "igsd_ig_teacher" in batch.batch and "igsd_ig_student" in batch.batch:
        ig_teacher = batch.batch["igsd_ig_teacher"].detach().float()
        ig_student = batch.batch["igsd_ig_student"].detach().float()
        if ig_teacher.shape != expected_shape or ig_student.shape != expected_shape:
            raise ValueError(
                "Paired-IG tensors must align with action-likelihood rows, got "
                f"{tuple(ig_teacher.shape)} and {tuple(ig_student.shape)} vs {tuple(expected_shape)}"
            )
        ig_delta_all = ig_teacher[:, 0] - ig_student[:, 0]
        paired_diagnostic_rows = valid_rows & torch.isfinite(ig_delta_all)
        paired_gap = query_gap[paired_diagnostic_rows]
        ig_delta = ig_delta_all[paired_diagnostic_rows]
        ig_positive = ig_delta.gt(margin)
        paired_likelihood_positive = paired_gap.gt(margin)
        both_positive = paired_likelihood_positive & ig_positive
        false_accept = paired_likelihood_positive & ~ig_positive
        missed_positive = ~paired_likelihood_positive & ig_positive
        both_nonpositive = ~paired_likelihood_positive & ~ig_positive
        metrics.update(
            {
                "igsd/action_likelihood_ig_diagnostic_valid_row_count": float(
                    paired_diagnostic_rows.sum().item()
                ),
                "igsd/action_likelihood_ig_diagnostic_invalid_row_count": float(
                    (valid_rows & ~torch.isfinite(ig_delta_all)).sum().item()
                ),
                "igsd/action_likelihood_ig_delta_pearson": _pearson_correlation(paired_gap, ig_delta),
                "igsd/action_likelihood_ig_sign_agreement_frac": scalar_mean(
                    (paired_likelihood_positive == ig_positive).float()
                ),
                "igsd/action_likelihood_ig_both_positive_frac": scalar_mean(both_positive.float()),
                "igsd/action_likelihood_ig_false_accept_frac": scalar_mean(false_accept.float()),
                "igsd/action_likelihood_ig_missed_positive_frac": scalar_mean(missed_positive.float()),
                "igsd/action_likelihood_ig_both_nonpositive_frac": scalar_mean(
                    both_nonpositive.float()
                ),
                "igsd/action_likelihood_ig_positive_precision": float(
                    both_positive.sum().float().div(paired_likelihood_positive.sum().clamp_min(1)).item()
                ),
                "igsd/action_likelihood_ig_positive_recall": float(
                    both_positive.sum().float().div(ig_positive.sum().clamp_min(1)).item()
                ),
            }
        )

    return metrics


def extract_search_queries(response_text: str, *, strict_schema: bool = False) -> list[str]:
    """Extract the first complete search call while preserving query slots."""

    for match in TOOL_CALL_PATTERN.finditer(response_text):
        try:
            call: Any = json.loads(match.group(1))
            if isinstance(call, list):
                if strict_schema:
                    continue
                call = call[0] if call else None
            if not isinstance(call, dict) or call.get("name") != "search":
                continue
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            query_list = arguments.get("query_list", [])
            if isinstance(query_list, list):
                if strict_schema and (
                    not query_list
                    or any(not isinstance(item, str) or not item.strip() for item in query_list)
                ):
                    continue
                queries = [item.strip() for item in query_list if isinstance(item, str) and item.strip()]
            else:
                if strict_schema:
                    continue
                queries = [str(query_list).strip()] if str(query_list).strip() else []
            if queries:
                return queries
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return []


def extract_first_search_query(response_text: str) -> str:
    """Legacy flattened wrapper used by single-query diagnostics."""

    return " ".join(extract_search_queries(response_text))


def _normalize_search_query_for_dedup(query: str) -> str:
    """Mirror ToolAgentLoop's duplicate-query normalization."""

    text = str(query).lower()
    punctuation = set(string.punctuation)
    text = "".join(character for character in text if character not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def build_teacher_prompt(
    raw_prompt: Any,
    failed_prefix_text: str,
    successful_queries: list[str],
    success_score: float,
    max_queries_per_tool_call: int = 1,
) -> list[dict[str, str]]:
    """Build privileged query-router input without GT or successful trace text."""

    prompt = raw_prompt.tolist() if isinstance(raw_prompt, np.ndarray) else raw_prompt
    prompt = [dict(message) for message in prompt] if isinstance(prompt, list) else [
        {"role": "user", "content": str(prompt)}
    ]
    query_lines = "\n".join(f"- {query}" for query in successful_queries if query.strip())
    max_queries = max(int(max_queries_per_tool_call), 1)
    query_instruction = (
        "with exactly one query"
        if max_queries == 1
        else f"with one search action containing at most {max_queries} queries in query_list"
    )
    instruction = (
        "You are the query-repair branch of a search agent. Propose exactly one next search action "
        "for the failed trajectory prefix below. A successful sibling's search queries and score are "
        "provided only as strategy hints. Do not answer the question, discuss the hint, or emit thought. "
        f"Output exactly one <tool_call> for the search tool {query_instruction}.\n\n"
        f"Failed trajectory prefix:\n{failed_prefix_text}\n\n"
        f"Successful sibling score: {success_score:.4f}\n"
        f"Successful sibling search queries:\n{query_lines or '- unavailable'}"
    )
    if prompt and prompt[0].get("role") == "system":
        prompt[0]["content"] = f"{prompt[0].get('content', '')}\n\n{instruction}"
    else:
        prompt.insert(0, {"role": "system", "content": instruction})
    return prompt


def clean_prompt_ids(batch: DataProto, sample_idx: int, pad_token_id: int) -> list[int]:
    prompt_ids = batch.batch["prompts"][sample_idx].tolist()
    start = 0
    while start < len(prompt_ids) and prompt_ids[start] == pad_token_id:
        start += 1
    return prompt_ids[start:]


def student_prefix_ids(
    batch: DataProto,
    sample_idx: int,
    turn: SearchTurn,
    pad_token_id: int,
) -> list[int]:
    """Prefix used for replacing the current search action.

    The prefix intentionally stops before ``turn.query_start``: the BC target is
    the teacher's replacement search action, so the student's original search
    action and its observation must not be included.
    """

    return clean_prompt_ids(batch, sample_idx, pad_token_id) + batch.batch["responses"][
        sample_idx, : turn.query_start
    ].tolist()


def _tool_response_text(documents: list[str]) -> str:
    result = "\n\n".join(str(doc) for doc in documents)
    return json.dumps({"result": result}, ensure_ascii=False)


def _tool_response_text_by_query(
    query_documents: list[list[str]], separator: str = "\n-*-*-\n"
) -> str:
    """Serialize per-query retrieval results using ASearcher separators."""

    chunks = [
        "\n\n".join(str(doc) for doc in documents if str(doc).strip())
        for documents in query_documents
    ]
    return json.dumps({"result": separator.join(chunks)}, ensure_ascii=False)


def _replace_tool_response_payload(tool_response_template_text: str, payload: str) -> str:
    """Replace tool content while preserving the trajectory's chat wrapper."""

    wrapper_pairs = (
        ("<tool_response>", "</tool_response>"),
        ("<|im_start|>tool", "<|im_end|>"),
        ("<|start_header_id|>tool<|end_header_id|>", "<|eot_id|>"),
    )
    for opening, closing in wrapper_pairs:
        opening_start = tool_response_template_text.find(opening)
        closing_start = tool_response_template_text.rfind(closing)
        if opening_start < 0 or closing_start < opening_start:
            continue
        content_start = opening_start + len(opening)
        # Preserve the chat template's newline immediately after the role tag.
        leading = "\n" if tool_response_template_text[content_start : content_start + 1] == "\n" else ""
        trailing = "\n" if tool_response_template_text[closing_start - 1 : closing_start] == "\n" else ""
        return (
            tool_response_template_text[:content_start]
            + leading
            + payload
            + trailing
            + tool_response_template_text[closing_start:]
        )

    payload_start = tool_response_template_text.find("{")
    payload_end = tool_response_template_text.rfind("}")
    if payload_start >= 0 and payload_end >= payload_start:
        candidate = tool_response_template_text[payload_start : payload_end + 1]
        try:
            parsed_candidate = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            parsed_candidate = None
        if isinstance(parsed_candidate, dict) and "result" in parsed_candidate:
            return (
                tool_response_template_text[:payload_start]
                + payload
                + tool_response_template_text[payload_end + 1 :]
            )

    # Last-resort Hermes wrapper for malformed or wrapper-free traces.
    return f"<tool_response>\n{payload}\n</tool_response>\n"


def _synthetic_tool_response_ids(
    documents: list[str],
    tool_response_template_text: str,
    tokenizer: Any,
    max_tool_response_tokens: int,
    tool_response_truncate_side: str,
) -> list[int]:
    """Encode synthetic docs with the same role wrappers as the real tool response."""

    payload = _tool_response_text(documents)
    payload = _truncate_tool_response_text(
        payload,
        tokenizer,
        max_tool_response_tokens,
        tool_response_truncate_side,
    )
    text = _replace_tool_response_payload(tool_response_template_text, payload)
    return tokenizer.encode(text, add_special_tokens=False)


def _synthetic_tool_response_ids_by_query(
    query_documents: list[list[str]],
    tool_response_template_text: str,
    tokenizer: Any,
    max_tool_response_tokens: int,
    tool_response_truncate_side: str,
    result_separator: str = "\n-*-*-\n",
) -> list[int]:
    """Encode multi-query retrieval evidence while retaining slot boundaries."""

    payload = _tool_response_text_by_query(query_documents, separator=result_separator)
    payload = _truncate_tool_response_text(
        payload,
        tokenizer,
        max_tool_response_tokens,
        tool_response_truncate_side,
    )
    text = _replace_tool_response_payload(tool_response_template_text, payload)
    return tokenizer.encode(text, add_special_tokens=False)


def _truncate_tool_response_text(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    truncate_side: str,
) -> str:
    """Mirror the agent-loop token truncation used for tool-role messages."""

    if not text or max_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text

    if truncate_side == "left":
        truncated = tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=True)
        return f"(truncated)...{truncated}"
    if truncate_side == "right":
        truncated = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
        return f"{truncated}...(truncated)"
    if truncate_side == "middle":
        left_n = max_tokens // 2
        right_n = max_tokens - left_n
        left = tokenizer.decode(token_ids[:left_n], skip_special_tokens=True)
        right = tokenizer.decode(token_ids[-right_n:], skip_special_tokens=True)
        return f"{left}...(truncated)...{right}"
    raise NotImplementedError(
        "IGSD M2 synthetic tool responses support left/middle/right truncation, "
        f"got {truncate_side!r}"
    )


def _sample_shared_docs(
    doc_pool: list[str],
    excluded: set[str],
    n_docs: int,
    rng: np.random.Generator,
) -> list[str]:
    candidates = [doc for doc in doc_pool if doc.strip() and doc.strip() not in excluded]
    if not candidates:
        return []
    size = min(max(n_docs, 1), len(candidates))
    indices = rng.choice(len(candidates), size=size, replace=False)
    return [candidates[int(idx)] for idx in np.atleast_1d(indices)]


def _dedup_docs(documents: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for doc in documents:
        text = str(doc).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def build_token_intervention_branch_specs(
    examples: list[dict[str, Any]],
    default_max_new_tokens: int,
    max_model_len: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Build exact token-prefix branch requests for G1 interventions.

    For a query with ``k`` value-token positions, the returned branch records
    contain ``B_0 ... B_{k-1}`` teacher-continuation branches and a terminal
    ``B_k=S`` record for the exact sampled student action.  ``B_q`` fixes the
    first ``q`` student query-value tokens and lets the teacher greedily
    continue from that exact token prefix.  The fixed prefix includes every
    original action token before the query-value span (including any thought
    or JSON scaffolding), so no text decode/re-tokenize step can shift the
    intervention index.

    ``max_model_len`` is treated as a hard prefix-preservation boundary. A
    branch that has no room for a generated token is marked invalid instead of
    truncating its fixed prefix; each backend may reserve an additional EOS
    slot and will still clamp the requested budget.
    """

    requests: list[dict[str, Any]] = []
    records_by_example: list[list[dict[str, Any]]] = []
    default_budget = max(int(default_max_new_tokens), 1)
    for example_idx, ex in enumerate(examples):
        student_queries = [
            str(query).strip()
            for query in ex.get("student_queries", [ex.get("student_query", "")])
            if str(query).strip()
        ]
        action_ids = [int(token_id) for token_id in ex.get("student_action_ids", [])]
        query_mask = [int(value) for value in ex.get("student_action_query_mask", [])]
        if len(action_ids) != len(query_mask):
            raise ValueError(
                "student action/query mask must align for token intervention: "
                f"{len(action_ids)=}, {len(query_mask)=}"
            )
        query_positions = [idx for idx, value in enumerate(query_mask) if value]
        teacher_prompt_ids = [int(token_id) for token_id in ex.get("teacher_prompt_ids", [])]
        records: list[dict[str, Any]] = []

        for branch_idx, query_position in enumerate(query_positions):
            # Fix the complete original action prefix immediately before the
            # intervened query token. For normal contiguous query masks this is
            # exactly the first ``branch_idx`` query tokens; for a rare
            # non-contiguous mask it also preserves intervening structure IDs.
            fixed_action_ids = action_ids[:query_position]
            prompt_ids = teacher_prompt_ids + fixed_action_ids
            record: dict[str, Any] = {
                "example_index": example_idx,
                "branch_index": branch_idx,
                "query_position": branch_idx,
                "action_query_position": query_position,
                "branch_kind": "teacher_continuation",
                "fixed_action_ids": fixed_action_ids,
                "prompt_length": len(prompt_ids) if teacher_prompt_ids else None,
                "branch_valid": False,
                "branch_ig_valid": False,
                "branch_query": "",
                "branch_queries": [],
                "branch_error": None,
                "request_index": None,
                "max_new_tokens": None,
            }
            if not teacher_prompt_ids:
                record["branch_error"] = "missing_teacher_prompt"
                records.append(record)
                continue
            if max_model_len > 0 and len(prompt_ids) >= max_model_len:
                record["branch_error"] = "prefix_reaches_max_model_len"
                records.append(record)
                continue
            budget = default_budget
            if max_model_len > 0:
                budget = min(budget, max(max_model_len - len(prompt_ids), 0))
            if budget <= 0:
                record["branch_error"] = "no_generation_budget"
                records.append(record)
                continue
            record["max_new_tokens"] = budget
            record["request_index"] = len(requests)
            requests.append(
                {
                    "prompt_ids": prompt_ids,
                    "sampling_params": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "repetition_penalty": 1.0,
                        "logprobs": False,
                        "max_tokens": budget,
                    },
                    "example_index": example_idx,
                    "branch_index": branch_idx,
                }
            )
            records.append(record)

        # B_k is the original sampled student action. It is scored by the
        # same paired-IG proxy but never sent to the teacher rollout server.
        records.append(
            {
                "example_index": example_idx,
                "branch_index": len(query_positions),
                "query_position": None,
                "branch_kind": "student_terminal",
                "fixed_action_ids": action_ids,
                "prompt_length": None,
                "branch_valid": bool(student_queries),
                "branch_ig_valid": False,
                "branch_query": " ".join(student_queries),
                "branch_queries": student_queries,
                "branch_error": None,
                "request_index": None,
                "max_new_tokens": None,
            }
        )
        records_by_example.append(records)
    return requests, records_by_example


def _stable_sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(min(value, 60.0), -60.0))))


def build_budgeted_local_candidate_branch_specs(
    examples: list[dict[str, Any]],
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    student_sample_log_probs: torch.Tensor,
    default_max_new_tokens: int,
    max_model_len: int,
    config: Any,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, float]]:
    """Plan and generate a bounded set of local teacher candidates.

    The cheap teacher/student log-prob pass classifies each query-value token as
    one of three states:

    * confirmation: teacher top-1 is the sampled student token and top-2 is
      not competitive; no environment branch is issued and a SDAR-style
      teacher/student log-prob gap supplies the token weight;
    * correction: teacher top-1 differs from the sampled token; top-1 receives
      priority for the environment budget;
    * ambiguity: teacher top-1/top-2 log-prob margin is small; after correction
      branches have been allocated, a position can receive both candidates.

    ``teacher_topk_*`` is produced from a teacher forward with the sampled
    student action as an additional candidate. Its final column is therefore
    the teacher log-probability of the sampled token. ``student_sample_log_probs``
    contains the corresponding student log-probability in its final column.
    The total requested teacher candidate count is bounded by
    ``floor(budget_ratio * k) <= k`` for each query with ``k`` query-value
    tokens. The exact student action is a separate shared terminal branch.
    """

    if teacher_topk_ids.ndim != 3 or teacher_topk_log_probs.shape != teacher_topk_ids.shape:
        raise ValueError(
            "Budgeted candidate teacher IDs/log-probs must be aligned rank-3 tensors, got "
            f"{teacher_topk_ids.shape=} and {teacher_topk_log_probs.shape=}"
        )
    if student_sample_log_probs.ndim != 3 or student_sample_log_probs.shape[:2] != teacher_topk_ids.shape[:2]:
        raise ValueError(
            "Budgeted candidate student log-probs must align with teacher positions, got "
            f"{student_sample_log_probs.shape=} and {teacher_topk_ids.shape=}"
        )
    if len(examples) != teacher_topk_ids.shape[0]:
        raise ValueError(
            "Budgeted candidate metadata and teacher target batch size must match, got "
            f"{len(examples)=} and {teacher_topk_ids.shape[0]=}"
        )
    if teacher_topk_ids.shape[-1] < 3 or teacher_topk_log_probs.shape[-1] < 3:
        raise ValueError(
            "Budgeted local candidates require teacher top-2 IDs plus the sampled student token log-prob"
        )
    if student_sample_log_probs.shape[-1] < 3:
        raise ValueError(
            "Budgeted local candidates require the sampled student token student log-prob"
        )

    budget_ratio = float(config.get("igsd_budgeted_candidate_budget_ratio", 1.0))
    ambiguity_margin = float(config.get("igsd_budgeted_ambiguity_log_margin", 0.5))
    confirmation_mode = str(config.get("igsd_budgeted_confirmation_mode", "soft")).lower()
    confirmation_beta = float(config.get("igsd_budgeted_confirmation_beta", 5.0))
    confirmation_margin = float(config.get("igsd_budgeted_confirmation_margin", 0.0))
    if not 0.0 <= budget_ratio <= 1.0:
        raise ValueError(f"Budgeted candidate ratio must be in [0, 1], got {budget_ratio}")
    if ambiguity_margin < 0.0:
        raise ValueError(f"Budgeted ambiguity margin must be non-negative, got {ambiguity_margin}")
    supported_confirmation_modes = {"soft", "positive_only", "none"}
    if confirmation_mode not in supported_confirmation_modes:
        raise ValueError(
            "Budgeted confirmation mode must be one of "
            f"{sorted(supported_confirmation_modes)}, got {confirmation_mode!r}"
        )

    teacher_topk_ids = teacher_topk_ids.detach().cpu()
    teacher_topk_log_probs = teacher_topk_log_probs.detach().float().cpu()
    student_sample_log_probs = student_sample_log_probs.detach().float().cpu()
    default_budget = max(int(default_max_new_tokens), 1)

    requests: list[dict[str, Any]] = []
    records_by_example: list[list[dict[str, Any]]] = []
    plans_by_example: list[list[dict[str, Any]]] = []
    total_query_tokens = 0
    total_budget = 0
    total_planned_candidates = 0
    total_confirmation = 0
    total_confirmation_positive_gap = 0
    total_confirmation_zero_gap = 0
    total_confirmation_negative_gap = 0
    total_confirmation_positive_weight = 0.0
    total_confirmation_zero_weight = 0.0
    total_confirmation_negative_weight = 0.0
    total_correction = 0
    total_ambiguity = 0
    total_budget_skipped = 0

    for example_idx, ex in enumerate(examples):
        student_queries = [
            str(query).strip()
            for query in ex.get("student_queries", [ex.get("student_query", "")])
            if str(query).strip()
        ]
        action_ids = [int(token_id) for token_id in ex.get("student_action_ids", [])]
        query_mask = [int(value) for value in ex.get("student_action_query_mask", [])]
        if len(action_ids) != len(query_mask):
            raise ValueError(
                "student action/query mask must align for budgeted candidate intervention: "
                f"{len(action_ids)=}, {len(query_mask)=}"
            )
        query_positions = [idx for idx, value in enumerate(query_mask) if value]
        teacher_prompt_ids = [int(token_id) for token_id in ex.get("teacher_prompt_ids", [])]
        if len(action_ids) > teacher_topk_ids.shape[1]:
            raise ValueError(
                "Teacher prefilter response width is shorter than the sampled action, got "
                f"{teacher_topk_ids.shape[1]=} and {len(action_ids)=}"
            )

        records: list[dict[str, Any]] = []
        plans: list[dict[str, Any]] = []
        correction_indices: list[int] = []
        ambiguous_indices: list[int] = []
        for query_idx, action_query_position in enumerate(query_positions):
            student_token_id = int(action_ids[action_query_position])
            top1_id = int(teacher_topk_ids[example_idx, action_query_position, 0].item())
            top2_id = int(teacher_topk_ids[example_idx, action_query_position, 1].item())
            top1_log_prob = float(teacher_topk_log_probs[example_idx, action_query_position, 0].item())
            top2_log_prob = float(teacher_topk_log_probs[example_idx, action_query_position, 1].item())
            teacher_student_log_prob = float(
                teacher_topk_log_probs[example_idx, action_query_position, -1].item()
            )
            student_student_log_prob = float(
                student_sample_log_probs[example_idx, action_query_position, -1].item()
            )
            finite = all(
                math.isfinite(value)
                for value in (
                    top1_log_prob,
                    top2_log_prob,
                    teacher_student_log_prob,
                    student_student_log_prob,
                )
            )
            teacher_margin = top1_log_prob - top2_log_prob if finite else float("nan")
            alternative_advantage = top1_log_prob - teacher_student_log_prob if finite else float("nan")
            ambiguous = bool(finite and teacher_margin <= ambiguity_margin)
            agrees = bool(finite and top1_id == student_token_id)
            if finite and agrees and not ambiguous:
                state = "confirmation"
                confirmation_signal = (
                    teacher_student_log_prob
                    - student_student_log_prob
                    - confirmation_margin
                )
                soft_confirmation_gate = _stable_sigmoid(
                    confirmation_beta * confirmation_signal
                )
                if confirmation_mode == "soft":
                    confirmation_gate = soft_confirmation_gate
                elif confirmation_mode == "positive_only":
                    confirmation_gate = (
                        soft_confirmation_gate if confirmation_signal > 0.0 else 0.0
                    )
                else:
                    confirmation_gate = 0.0
                total_confirmation += 1
                if confirmation_signal > 0.0:
                    total_confirmation_positive_gap += 1
                    total_confirmation_positive_weight += confirmation_gate
                elif confirmation_signal < 0.0:
                    total_confirmation_negative_gap += 1
                    total_confirmation_negative_weight += confirmation_gate
                else:
                    total_confirmation_zero_gap += 1
                    total_confirmation_zero_weight += confirmation_gate
            elif finite and not agrees:
                state = "correction"
                confirmation_gate = 0.0
                confirmation_signal = 0.0
                correction_indices.append(query_idx)
                if ambiguous:
                    ambiguous_indices.append(query_idx)
                total_correction += 1
            elif finite:
                state = "ambiguity"
                confirmation_gate = 0.0
                confirmation_signal = 0.0
                ambiguous_indices.append(query_idx)
                total_ambiguity += 1
            else:
                state = "invalid_prefilter"
                confirmation_gate = 0.0
                confirmation_signal = 0.0
            plans.append(
                {
                    "query_position": query_idx,
                    "action_query_position": action_query_position,
                    "student_token_id": student_token_id,
                    "teacher_top1_id": top1_id,
                    "teacher_top2_id": top2_id,
                    "teacher_top1_log_prob": top1_log_prob,
                    "teacher_top2_log_prob": top2_log_prob,
                    "teacher_student_log_prob": teacher_student_log_prob,
                    "student_student_log_prob": student_student_log_prob,
                    "teacher_student_log_gap": teacher_student_log_prob - student_student_log_prob,
                    "confirmation_signal": confirmation_signal,
                    "teacher_margin": teacher_margin,
                    "alternative_advantage": alternative_advantage,
                    "state": state,
                    "confirmation_gate": confirmation_gate,
                    "top1_branch_index": None,
                    "top2_branch_index": None,
                }
            )

        query_token_count = len(query_positions)
        branch_budget = int(math.floor(budget_ratio * query_token_count + 1e-12))
        total_query_tokens += query_token_count
        total_budget += branch_budget
        remaining_budget = branch_budget

        # First cover explicit teacher/student corrections. The score is the
        # teacher's within-distribution preference for its top-1 over student.
        for query_idx in sorted(
            correction_indices,
            key=lambda idx: float(plans[idx]["alternative_advantage"]),
            reverse=True,
        ):
            if remaining_budget <= 0:
                break
            plans[query_idx]["select_top1"] = True
            remaining_budget -= 1

        # Upgrade selected ambiguous corrections with top-2 before allocating
        # a two-branch ambiguity probe to agreement positions. This keeps the
        # correction path covered whenever the hard budget is saturated.
        selected_ambiguous_corrections = [
            idx
            for idx in ambiguous_indices
            if plans[idx]["state"] == "correction" and plans[idx].get("select_top1", False)
        ]
        for query_idx in sorted(
            selected_ambiguous_corrections,
            key=lambda idx: float(plans[idx]["teacher_margin"]),
        ):
            if remaining_budget <= 0:
                break
            plans[query_idx]["select_top2"] = True
            remaining_budget -= 1

        agreement_ambiguities = [
            idx for idx in ambiguous_indices if plans[idx]["state"] == "ambiguity"
        ]
        for query_idx in sorted(
            agreement_ambiguities,
            key=lambda idx: float(plans[idx]["teacher_margin"]),
        ):
            if remaining_budget < 2:
                break
            plans[query_idx]["select_top1"] = True
            plans[query_idx]["select_top2"] = True
            remaining_budget -= 2

        def add_candidate_branch(plan: dict[str, Any], rank: int) -> None:
            action_query_position = int(plan["action_query_position"])
            if rank == 1:
                fixed_action_ids = action_ids[:action_query_position]
                candidate_token_id = int(plan["teacher_top1_id"])
                expected_first_generated_token = candidate_token_id
            elif rank == 2:
                candidate_token_id = int(plan["teacher_top2_id"])
                fixed_action_ids = action_ids[:action_query_position] + [candidate_token_id]
                expected_first_generated_token = None
            else:
                raise ValueError(f"Unsupported budgeted local candidate rank {rank}")
            prompt_ids = teacher_prompt_ids + fixed_action_ids
            record: dict[str, Any] = {
                "example_index": example_idx,
                "branch_index": len(records),
                "query_position": int(plan["query_position"]),
                "action_query_position": action_query_position,
                "branch_kind": f"budgeted_teacher_top{rank}",
                "candidate_rank": rank,
                "candidate_token_id": candidate_token_id,
                "expected_first_generated_token": expected_first_generated_token,
                "fixed_action_ids": fixed_action_ids,
                "prompt_length": len(prompt_ids) if teacher_prompt_ids else None,
                "branch_valid": False,
                "branch_ig_valid": False,
                "branch_query": "",
                "branch_queries": [],
                "branch_error": None,
                "request_index": None,
                "max_new_tokens": None,
            }
            plan[f"top{rank}_branch_index"] = int(record["branch_index"])
            if not teacher_prompt_ids:
                record["branch_error"] = "missing_teacher_prompt"
                records.append(record)
                return
            if max_model_len > 0 and len(prompt_ids) >= max_model_len:
                record["branch_error"] = "prefix_reaches_max_model_len"
                records.append(record)
                return
            branch_max_new_tokens = default_budget
            if max_model_len > 0:
                branch_max_new_tokens = min(branch_max_new_tokens, max(max_model_len - len(prompt_ids), 0))
            if branch_max_new_tokens <= 0:
                record["branch_error"] = "no_generation_budget"
                records.append(record)
                return
            record["max_new_tokens"] = branch_max_new_tokens
            record["request_index"] = len(requests)
            requests.append(
                {
                    "prompt_ids": prompt_ids,
                    "sampling_params": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "repetition_penalty": 1.0,
                        "logprobs": False,
                        "max_tokens": branch_max_new_tokens,
                    },
                    "example_index": example_idx,
                    "branch_index": int(record["branch_index"]),
                }
            )
            records.append(record)

        for plan in plans:
            if plan.get("select_top1", False):
                add_candidate_branch(plan, rank=1)
            if plan.get("select_top2", False):
                add_candidate_branch(plan, rank=2)
            if plan["state"] != "confirmation" and not plan.get("select_top1", False):
                plan["state"] = "budget_skip" if plan["state"] != "invalid_prefilter" else plan["state"]
                total_budget_skipped += int(plan["state"] == "budget_skip")

        # The shared student terminal is scored once, regardless of how many
        # teacher candidate branches fit in the budget.
        records.append(
            {
                "example_index": example_idx,
                "branch_index": len(records),
                "query_position": None,
                "action_query_position": None,
                "branch_kind": "student_terminal",
                "fixed_action_ids": action_ids,
                "prompt_length": None,
                "branch_valid": bool(student_queries),
                "branch_ig_valid": False,
                "branch_query": " ".join(student_queries),
                "branch_queries": student_queries,
                "branch_error": None,
                "request_index": None,
                "max_new_tokens": None,
            }
        )
        total_planned_candidates += sum(
            int(plan.get("select_top1", False)) + int(plan.get("select_top2", False))
            for plan in plans
        )
        if sum(
            int(plan.get("select_top1", False)) + int(plan.get("select_top2", False))
            for plan in plans
        ) > branch_budget:
            raise AssertionError("Budgeted local candidate planner exceeded its per-query teacher branch budget")
        records_by_example.append(records)
        plans_by_example.append(plans)

    if total_planned_candidates > total_budget:
        raise AssertionError("Budgeted local candidate planner exceeded its batch teacher branch budget")
    metrics = {
        "igsd/budgeted_candidate_enabled": 1.0,
        "igsd/budgeted_candidate_query_token_count": float(total_query_tokens),
        "igsd/budgeted_candidate_teacher_branch_budget": float(total_budget),
        "igsd/budgeted_candidate_teacher_branch_planned_count": float(total_planned_candidates),
        "igsd/budgeted_candidate_teacher_branch_budget_utilization": float(
            total_planned_candidates / max(total_budget, 1)
        ),
        "igsd/budgeted_candidate_confirmation_token_count": float(total_confirmation),
        "igsd/budgeted_candidate_confirmation_mode_is_soft": float(
            confirmation_mode == "soft"
        ),
        "igsd/budgeted_candidate_confirmation_mode_is_positive_only": float(
            confirmation_mode == "positive_only"
        ),
        "igsd/budgeted_candidate_confirmation_mode_is_none": float(
            confirmation_mode == "none"
        ),
        "igsd/budgeted_candidate_confirmation_positive_gap_token_count": float(
            total_confirmation_positive_gap
        ),
        "igsd/budgeted_candidate_confirmation_zero_gap_token_count": float(
            total_confirmation_zero_gap
        ),
        "igsd/budgeted_candidate_confirmation_negative_gap_token_count": float(
            total_confirmation_negative_gap
        ),
        "igsd/budgeted_candidate_confirmation_positive_gap_frac": float(
            total_confirmation_positive_gap / max(total_confirmation, 1)
        ),
        "igsd/budgeted_candidate_confirmation_zero_gap_frac": float(
            total_confirmation_zero_gap / max(total_confirmation, 1)
        ),
        "igsd/budgeted_candidate_confirmation_negative_gap_frac": float(
            total_confirmation_negative_gap / max(total_confirmation, 1)
        ),
        "igsd/budgeted_candidate_confirmation_positive_weight_sum": float(
            total_confirmation_positive_weight
        ),
        "igsd/budgeted_candidate_confirmation_zero_weight_sum": float(
            total_confirmation_zero_weight
        ),
        "igsd/budgeted_candidate_confirmation_negative_weight_sum": float(
            total_confirmation_negative_weight
        ),
        "igsd/budgeted_candidate_confirmation_weight_sum": float(
            total_confirmation_positive_weight
            + total_confirmation_zero_weight
            + total_confirmation_negative_weight
        ),
        "igsd/budgeted_candidate_correction_token_count": float(total_correction),
        "igsd/budgeted_candidate_ambiguity_token_count": float(total_ambiguity),
        "igsd/budgeted_candidate_budget_skip_token_count": float(total_budget_skipped),
    }
    return requests, records_by_example, plans_by_example, metrics


def _sampled_pair_distribution_diagnostics(
    teacher_token_id: int,
    sampled_token_id: int,
    teacher_token_log_prob: float,
    teacher_sampled_log_prob: float,
    student_token_log_prob: float,
    student_sampled_log_prob: float,
) -> dict[str, float | None]:
    """Return full-mass and pair-conditional diagnostics without changing routing."""

    def safe_exp(value: float) -> float:
        try:
            return math.exp(value)
        except OverflowError:
            return float("inf")

    pair_logs = (
        teacher_token_log_prob,
        teacher_sampled_log_prob,
        student_token_log_prob,
        student_sampled_log_prob,
    )
    if not all(math.isfinite(value) for value in pair_logs):
        return {
            "teacher_pair_log_mass": None,
            "teacher_pair_mass": None,
            "student_pair_log_mass": None,
            "student_pair_mass": None,
            "teacher_pair_teacher_token_prob": None,
            "student_pair_teacher_token_prob": None,
            "prefilter_pair_jsd": None,
            "prefilter_pair_jsd_student_logit_grad": None,
            "prefilter_pair_jsd_student_logit_grad_abs": None,
        }

    # A duplicated candidate is a one-event support, not two copies of the same
    # probability mass. It is skipped by the protocol, so pair-conditional JSD
    # diagnostics are intentionally undefined while the unique event mass remains useful.
    if teacher_token_id == sampled_token_id:
        return {
            "teacher_pair_log_mass": teacher_token_log_prob,
            "teacher_pair_mass": safe_exp(teacher_token_log_prob),
            "student_pair_log_mass": student_token_log_prob,
            "student_pair_mass": safe_exp(student_token_log_prob),
            "teacher_pair_teacher_token_prob": None,
            "student_pair_teacher_token_prob": None,
            "prefilter_pair_jsd": None,
            "prefilter_pair_jsd_student_logit_grad": None,
            "prefilter_pair_jsd_student_logit_grad_abs": None,
        }

    teacher_pair_log_mass = float(np.logaddexp(teacher_token_log_prob, teacher_sampled_log_prob))
    student_pair_log_mass = float(np.logaddexp(student_token_log_prob, student_sampled_log_prob))
    teacher_prob = math.exp(teacher_token_log_prob - teacher_pair_log_mass)
    student_prob = math.exp(student_token_log_prob - student_pair_log_mass)
    mixture_prob = 0.5 * (teacher_prob + student_prob)

    def binary_entropy(probability: float) -> float:
        if probability <= 0.0 or probability >= 1.0:
            return 0.0
        return -probability * math.log(probability) - (1.0 - probability) * math.log1p(-probability)

    pair_jsd = max(
        binary_entropy(mixture_prob)
        - 0.5 * binary_entropy(teacher_prob)
        - 0.5 * binary_entropy(student_prob),
        0.0,
    )
    # Exact derivative of Bernoulli JSD with respect to the student's
    # teacher-vs-sampled logit. Its magnitude is a cheap proxy for how much the
    # selected pair objective can move the student at the prefilter checkpoint.
    if student_prob <= 0.0 or student_prob >= 1.0:
        student_logit_grad = 0.0
    else:
        eps = np.finfo(np.float64).eps
        clipped_mixture = min(max(mixture_prob, eps), 1.0 - eps)
        mixture_log_odds = math.log(clipped_mixture) - math.log1p(-clipped_mixture)
        student_log_odds = student_token_log_prob - student_sampled_log_prob
        student_logit_grad = (
            0.5
            * student_prob
            * (1.0 - student_prob)
            * (student_log_odds - mixture_log_odds)
        )
    return {
        "teacher_pair_log_mass": teacher_pair_log_mass,
        "teacher_pair_mass": safe_exp(teacher_pair_log_mass),
        "student_pair_log_mass": student_pair_log_mass,
        "student_pair_mass": safe_exp(student_pair_log_mass),
        "teacher_pair_teacher_token_prob": teacher_prob,
        "student_pair_teacher_token_prob": student_prob,
        "prefilter_pair_jsd": pair_jsd,
        "prefilter_pair_jsd_student_logit_grad": student_logit_grad,
        "prefilter_pair_jsd_student_logit_grad_abs": abs(student_logit_grad),
    }


def build_sampled_action_pair_branch_specs(
    examples: list[dict[str, Any]],
    teacher_top1_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_top1_ids: torch.Tensor,
    student_log_probs: torch.Tensor,
    default_max_new_tokens: int,
    max_model_len: int,
    config: Any,
    global_step: int = 0,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, float]]:
    """Plan sampled-action anchored teacher/student continuation pairs.

    Each eligible query position has three explicitly separated tokens:

    ``s``
        The token sampled by the student rollout and used as the behavioral
        reference action.
    ``t``
        The hindsight teacher top-1 proposal.
    ``u``
        The unprivileged student top-1 token, retained for diagnostics.

    When the per-query verification budget is exhausted, candidates are
    prioritized by the magnitude of the teacher-vs-reference log-odds shift,

    ``[log p(t) - log p(s)] - [log q(t) - log q(s)]``.

    Every finite distinct pair with a non-negligible shift is eligible,
    irrespective of sign. The executed environment gain determines acceptance.
    Every selected pair atomically consumes two logical branch-budget units.

    ``environment_ig`` materializes continuation branches: ``greedy_pair``
    creates both complete requests from the unprivileged student prefix, while
    ``fixed_s`` creates only the teacher request and reuses one shared original
    sampled terminal. ``likelihood_gap`` and ``constant`` keep the same routed
    pair metadata but intentionally create no environment branches.
    """

    if teacher_top1_ids.ndim == 3 and teacher_top1_ids.shape[-1] == 1:
        teacher_top1_ids = teacher_top1_ids.squeeze(-1)
    if student_top1_ids.ndim == 3 and student_top1_ids.shape[-1] == 1:
        student_top1_ids = student_top1_ids.squeeze(-1)
    if teacher_top1_ids.ndim != 2 or student_top1_ids.ndim != 2:
        raise ValueError(
            "Sampled-action pair top-1 IDs must have shape [batch, response], got "
            f"{teacher_top1_ids.shape=} and {student_top1_ids.shape=}"
        )
    if teacher_top1_ids.shape != student_top1_ids.shape:
        raise ValueError(
            "Sampled-action pair teacher/student top-1 IDs must align, got "
            f"{teacher_top1_ids.shape=} and {student_top1_ids.shape=}"
        )
    expected_teacher_log_shape = (*teacher_top1_ids.shape, 2)
    expected_student_log_shape = (*student_top1_ids.shape, 3)
    if teacher_log_probs.shape != expected_teacher_log_shape:
        raise ValueError(
            "Sampled-action pair teacher log-probs must contain [top1, reference] columns, got "
            f"{teacher_log_probs.shape=} expected {expected_teacher_log_shape}"
        )
    if student_log_probs.shape != expected_student_log_shape:
        raise ValueError(
            "Sampled-action pair student log-probs must contain [top1, teacher, reference] columns, got "
            f"{student_log_probs.shape=} expected {expected_student_log_shape}"
        )
    if len(examples) != teacher_top1_ids.shape[0]:
        raise ValueError(
            "Sampled-action pair metadata and prefilter batch must align, got "
            f"{len(examples)=} and {teacher_top1_ids.shape[0]=}"
        )

    budget_ratio = float(config.get("igsd_sampled_pair_budget_ratio", 1.0))
    log_odds_epsilon = float(config.get("igsd_sampled_pair_log_odds_epsilon", 1e-6))
    reference_mode = str(config.get("igsd_sampled_pair_reference_mode", "greedy_pair")).lower()
    gate_source = str(
        config.get("igsd_sampled_pair_gate_source", "environment_ig")
    ).lower()
    audit_all_eligible = bool(
        config.get("igsd_sampled_pair_audit_all_eligible", False)
    )
    candidate_mode = str(
        config.get("igsd_sampled_pair_candidate_mode", "sampled_action")
    ).lower()
    if reference_mode not in {"greedy_pair", "fixed_s"}:
        raise ValueError(
            "Sampled-action pair reference mode must be 'greedy_pair' or 'fixed_s', "
            f"got {reference_mode!r}"
        )
    if gate_source not in {"environment_ig", "likelihood_gap", "constant"}:
        raise ValueError(
            "Sampled-action pair gate source must be 'environment_ig', "
            f"'likelihood_gap', or 'constant', got {gate_source!r}"
        )
    if int(global_step) < 0:
        raise ValueError(f"Sampled-action pair global_step must be non-negative, got {global_step}")
    if candidate_mode not in {"sampled_action", "student_top1"}:
        raise ValueError(
            "Sampled-action pair candidate mode must be 'sampled_action' or 'student_top1', "
            f"got {candidate_mode!r}"
        )
    if candidate_mode == "student_top1" and reference_mode != "greedy_pair":
        raise ValueError(
            "student_top1 modal-pair verification requires reference_mode='greedy_pair'; "
            "fixed_s would compare against the original sampled query instead of u_i"
        )
    if gate_source == "likelihood_gap" and candidate_mode != "sampled_action":
        raise ValueError(
            "likelihood_gap sampled-pair gating requires candidate_mode='sampled_action'"
        )
    max_budget_ratio = 2.0 if audit_all_eligible else 1.0
    if not math.isfinite(budget_ratio) or not 0.0 <= budget_ratio <= max_budget_ratio:
        raise ValueError(
            "Sampled-action pair budget ratio must be finite and in "
            f"[0, {max_budget_ratio:g}] when audit_all_eligible={audit_all_eligible}, "
            f"got {budget_ratio}"
        )
    if not math.isfinite(log_odds_epsilon) or log_odds_epsilon < 0.0:
        raise ValueError(
            "Sampled-action pair log-odds epsilon must be finite and non-negative, got "
            f"{log_odds_epsilon}"
        )

    teacher_top1_ids = teacher_top1_ids.detach().cpu().long()
    student_top1_ids = student_top1_ids.detach().cpu().long()
    teacher_log_probs = teacher_log_probs.detach().float().cpu()
    student_log_probs = student_log_probs.detach().float().cpu()
    default_budget = max(int(default_max_new_tokens), 1)

    requests: list[dict[str, Any]] = []
    records_by_example: list[list[dict[str, Any]]] = []
    plans_by_example: list[list[dict[str, Any]]] = []
    total_query_tokens = 0
    total_branch_budget = 0
    total_eligible = 0
    total_selected_pairs = 0
    total_normal_pair_capacity = 0
    total_budget_skipped = 0
    all_log_odds_shifts: list[float] = []
    identity_counts = {
        "all_equal": 0,
        "teacher_equals_student_top1_only": 0,
        "teacher_equals_sampled_only": 0,
        "student_top1_equals_sampled_only": 0,
        "all_distinct": 0,
    }
    prefilter_state_counts: dict[str, int] = {}
    diagnostic_values: dict[str, list[float]] = {
        "student_top1_log_prob": [],
        "student_top1_prob": [],
        "teacher_pair_log_mass": [],
        "teacher_pair_mass": [],
        "student_pair_log_mass": [],
        "student_pair_mass": [],
        "teacher_pair_teacher_token_prob": [],
        "student_pair_teacher_token_prob": [],
        "prefilter_pair_jsd": [],
        "prefilter_pair_jsd_student_logit_grad": [],
        "prefilter_pair_jsd_student_logit_grad_abs": [],
        "teacher_top1_minus_reference_log_prob": [],
        "student_top1_minus_reference_log_prob": [],
        "teacher_top1_minus_sampled_log_prob": [],
        "student_top1_minus_sampled_log_prob": [],
        "student_top1_minus_teacher_log_prob": [],
    }

    def increment(mapping: dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1

    for example_idx, ex in enumerate(examples):
        student_queries = [
            str(query).strip()
            for query in ex.get("student_queries", [ex.get("student_query", "")])
            if str(query).strip()
        ]
        action_ids = [int(token_id) for token_id in ex.get("student_action_ids", [])]
        query_mask = [int(value) for value in ex.get("student_action_query_mask", [])]
        prefix_ids = [int(token_id) for token_id in ex.get("prefix_ids", [])]
        if len(action_ids) != len(query_mask):
            raise ValueError(
                "Sampled-action pair student action/query mask must align, got "
                f"{len(action_ids)=} and {len(query_mask)=}"
            )
        if len(action_ids) > teacher_top1_ids.shape[1]:
            raise ValueError(
                "Sampled-action pair prefilter response is shorter than the student action, got "
                f"{teacher_top1_ids.shape[1]=} and {len(action_ids)=}"
            )
        query_positions = [idx for idx, value in enumerate(query_mask) if value]
        plans: list[dict[str, Any]] = []
        eligible_indices: list[int] = []
        for query_idx, action_query_position in enumerate(query_positions):
            sampled_id = int(action_ids[action_query_position])
            teacher_id = int(teacher_top1_ids[example_idx, action_query_position].item())
            student_top1_id = int(student_top1_ids[example_idx, action_query_position].item())
            teacher_top1_log_prob = float(
                teacher_log_probs[example_idx, action_query_position, 0].item()
            )
            teacher_reference_log_prob = float(
                teacher_log_probs[example_idx, action_query_position, 1].item()
            )
            student_top1_log_prob = float(
                student_log_probs[example_idx, action_query_position, 0].item()
            )
            student_teacher_log_prob = float(
                student_log_probs[example_idx, action_query_position, 1].item()
            )
            student_reference_log_prob = float(
                student_log_probs[example_idx, action_query_position, 2].item()
            )
            reference_id = sampled_id if candidate_mode == "sampled_action" else student_top1_id
            # These aliases are meaningful only for sampled-action
            # mode. Modal mode has no sampled-token log-prob in the prefilter
            # candidate columns; its target uses the explicit reference fields.
            teacher_sample_log_prob = (
                teacher_reference_log_prob if candidate_mode == "sampled_action" else None
            )
            student_sample_log_prob = (
                student_reference_log_prob if candidate_mode == "sampled_action" else None
            )
            finite = all(
                math.isfinite(value)
                for value in (
                    teacher_top1_log_prob,
                    teacher_reference_log_prob,
                    student_teacher_log_prob,
                    student_reference_log_prob,
                )
            )
            log_odds_shift = (
                teacher_top1_log_prob
                - teacher_reference_log_prob
                - student_teacher_log_prob
                + student_reference_log_prob
                if finite
                else float("nan")
            )
            if math.isfinite(log_odds_shift):
                all_log_odds_shifts.append(log_odds_shift)
            if teacher_id == sampled_id == student_top1_id:
                identity_state = "all_equal"
                identity_counts["all_equal"] += 1
            elif teacher_id == student_top1_id and teacher_id != sampled_id:
                identity_state = "teacher_equals_student_top1_only"
                identity_counts["teacher_equals_student_top1_only"] += 1
            elif teacher_id == sampled_id and teacher_id != student_top1_id:
                identity_state = "teacher_equals_sampled_only"
                identity_counts["teacher_equals_sampled_only"] += 1
            elif student_top1_id == sampled_id and student_top1_id != teacher_id:
                identity_state = "student_top1_equals_sampled_only"
                identity_counts["student_top1_equals_sampled_only"] += 1
            else:
                identity_state = "all_distinct"
                identity_counts["all_distinct"] += 1

            distribution_diagnostics = _sampled_pair_distribution_diagnostics(
                teacher_id,
                reference_id,
                teacher_top1_log_prob,
                teacher_reference_log_prob,
                student_teacher_log_prob,
                student_reference_log_prob,
            )
            reference_margins = {
                "teacher_top1_minus_reference_log_prob": (
                    teacher_top1_log_prob - teacher_reference_log_prob
                ),
                "student_top1_minus_reference_log_prob": (
                    student_top1_log_prob - student_reference_log_prob
                ),
                "student_top1_minus_teacher_log_prob": (
                    student_top1_log_prob - student_teacher_log_prob
                ),
            }
            if candidate_mode == "sampled_action":
                reference_margins.update(
                    {
                        "teacher_top1_minus_sampled_log_prob": reference_margins[
                            "teacher_top1_minus_reference_log_prob"
                        ],
                        "student_top1_minus_sampled_log_prob": reference_margins[
                            "student_top1_minus_reference_log_prob"
                        ],
                    }
                )
            student_top1_prob = (
                math.exp(student_top1_log_prob)
                if math.isfinite(student_top1_log_prob) and student_top1_log_prob <= 0.0
                else None
            )
            if math.isfinite(student_top1_log_prob):
                diagnostic_values["student_top1_log_prob"].append(student_top1_log_prob)
            if student_top1_prob is not None:
                diagnostic_values["student_top1_prob"].append(student_top1_prob)
            for name, value in distribution_diagnostics.items():
                if value is not None and math.isfinite(float(value)):
                    diagnostic_values[name].append(float(value))
            for name, value in reference_margins.items():
                if math.isfinite(value):
                    diagnostic_values[name].append(value)

            if not finite:
                state = "invalid_prefilter"
            elif teacher_id == reference_id:
                state = (
                    "teacher_equals_sampled_skip"
                    if candidate_mode == "sampled_action"
                    else "teacher_equals_student_top1_skip"
                )
            elif not is_eligible_disagreement(
                teacher_id, reference_id, log_odds_shift, epsilon=log_odds_epsilon
            ):
                state = "near_zero_disagreement"
            else:
                state = "eligible"
                eligible_indices.append(query_idx)
                total_eligible += 1
            increment(prefilter_state_counts, state)
            plans.append(
                {
                    "query_position": query_idx,
                    "action_query_position": action_query_position,
                    "sampled_token_id": sampled_id,
                    "reference_token_id": reference_id,
                    "teacher_top1_id": teacher_id,
                    "student_top1_id": student_top1_id,
                    "teacher_top1_log_prob": teacher_top1_log_prob,
                    "teacher_sample_log_prob": teacher_sample_log_prob,
                    "teacher_reference_log_prob": teacher_reference_log_prob,
                    "student_top1_log_prob": student_top1_log_prob,
                    "student_top1_prob": student_top1_prob,
                    "student_teacher_log_prob": student_teacher_log_prob,
                    "student_sample_log_prob": student_sample_log_prob,
                    "student_reference_log_prob": student_reference_log_prob,
                    "pair_log_odds_shift": log_odds_shift,
                    "pair_reference_mode": candidate_mode,
                    "pair_gate_source": gate_source,
                    "budget_priority": abs(log_odds_shift),
                    "selection_rank": None,
                    "budget_selected": False,
                    **distribution_diagnostics,
                    **reference_margins,
                    "identity_state": identity_state,
                    "state": state,
                    "teacher_branch_index": None,
                    "sampled_branch_index": None,
                    "teacher_branch_valid": False,
                    "sampled_branch_valid": False,
                }
            )

        query_token_count = len(query_positions)
        branch_budget = int(math.floor(budget_ratio * query_token_count + 1e-12))
        if audit_all_eligible:
            branch_budget = max(branch_budget, 2 * len(eligible_indices))
        total_query_tokens += query_token_count
        total_branch_budget += branch_budget
        remaining_budget = branch_budget

        # A pair is atomic: never spend one branch without its matched control.
        ordered_eligible_indices = sorted(
            eligible_indices,
            key=lambda idx: float(plans[idx]["budget_priority"]),
            reverse=True,
        )
        normal_branch_budget = int(
            math.floor(min(budget_ratio, 1.0) * query_token_count + 1e-12)
        )
        normal_pair_capacity = normal_branch_budget // 2
        total_normal_pair_capacity += normal_pair_capacity
        for selection_rank, query_idx in enumerate(ordered_eligible_indices):
            plans[query_idx]["selection_rank"] = int(selection_rank)
            plans[query_idx]["budget_selected"] = bool(selection_rank < normal_pair_capacity)
        for query_idx in ordered_eligible_indices:
            if remaining_budget < 2:
                plans[query_idx]["state"] = "budget_skipped"
                total_budget_skipped += 1
                continue
            plans[query_idx]["selected"] = True
            remaining_budget -= 2
            total_selected_pairs += 1

        records: list[dict[str, Any]] = []

        def add_pair_branch(plan: dict[str, Any], side: str, candidate_id: int) -> None:
            action_query_position = int(plan["action_query_position"])
            fixed_action_ids = action_ids[:action_query_position] + [int(candidate_id)]
            prompt_ids = prefix_ids + fixed_action_ids
            branch_index = len(records)
            record: dict[str, Any] = {
                "example_index": example_idx,
                "branch_index": branch_index,
                "query_position": int(plan["query_position"]),
                "action_query_position": action_query_position,
                "branch_kind": (
                    "sampled_pair_teacher" if side == "teacher" else "sampled_pair_student"
                ),
                "pair_side": side,
                "candidate_token_id": int(candidate_id),
                "sampled_token_id": int(plan["sampled_token_id"]),
                "reference_token_id": int(plan["reference_token_id"]),
                "teacher_top1_id": int(plan["teacher_top1_id"]),
                "student_top1_id": int(plan["student_top1_id"]),
                "pair_log_odds_shift": float(plan["pair_log_odds_shift"]),
                "fixed_action_ids": fixed_action_ids,
                "prompt_length": len(prompt_ids) if prefix_ids else None,
                "branch_valid": False,
                "branch_ig_valid": False,
                "branch_query": "",
                "branch_queries": [],
                "branch_error": None,
                "request_index": None,
                "max_new_tokens": None,
                # The candidate token is already in the prompt. The rollout
                # output is therefore the student continuation only.
                "expected_first_generated_token": None,
            }
            plan_key = "teacher_branch_index" if side == "teacher" else "sampled_branch_index"
            plan[plan_key] = branch_index
            if not prefix_ids:
                record["branch_error"] = "missing_student_prefix"
                records.append(record)
                return
            if max_model_len > 0 and len(prompt_ids) >= max_model_len:
                record["branch_error"] = "prefix_reaches_max_model_len"
                records.append(record)
                return
            budget = default_budget
            if max_model_len > 0:
                budget = min(budget, max(max_model_len - len(prompt_ids), 0))
            if budget <= 0:
                record["branch_error"] = "no_generation_budget"
                records.append(record)
                return
            record["max_new_tokens"] = budget
            record["request_index"] = len(requests)
            requests.append(
                {
                    "prompt_ids": prompt_ids,
                    "sampling_params": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "repetition_penalty": 1.0,
                        "logprobs": False,
                        "max_tokens": budget,
                    },
                    "example_index": example_idx,
                    "branch_index": branch_index,
                }
            )
            records.append(record)

        selected_plans = [plan for plan in plans if plan.get("selected", False)]
        for plan in plans:
            if plan.get("selected", False):
                if gate_source == "environment_ig":
                    add_pair_branch(plan, "teacher", int(plan["teacher_top1_id"]))
                    if reference_mode == "greedy_pair":
                        add_pair_branch(plan, "student", int(plan["reference_token_id"]))
            elif plan.get("state") == "eligible":
                plan["state"] = "budget_skipped"
                total_budget_skipped += 1
        if gate_source == "environment_ig" and reference_mode == "fixed_s" and selected_plans:
            terminal_branch_index = len(records)
            records.append(
                {
                    "example_index": example_idx,
                    "branch_index": terminal_branch_index,
                    "query_position": None,
                    "action_query_position": None,
                    "branch_kind": "student_terminal",
                    "pair_side": "fixed_s_reference",
                    "fixed_action_ids": list(action_ids),
                    "prompt_length": len(prefix_ids) if prefix_ids else None,
                    "branch_valid": bool(student_queries),
                    "branch_ig_valid": False,
                    "branch_query": " ".join(student_queries),
                    "branch_queries": student_queries,
                    "branch_error": None,
                    "request_index": None,
                    "max_new_tokens": None,
                }
            )
            for plan in selected_plans:
                plan["sampled_branch_index"] = terminal_branch_index
        records_by_example.append(records)
        plans_by_example.append(plans)

    final_state_counts: dict[str, int] = {}
    for plans in plans_by_example:
        for plan in plans:
            final_state = "selected" if bool(plan.get("selected", False)) else str(plan.get("state", "unknown"))
            plan["final_state"] = final_state
            increment(final_state_counts, final_state)

    metrics = {
        "igsd/sampled_pair_enabled": 1.0,
        "igsd/sampled_pair_query_token_count": float(total_query_tokens),
        "igsd/sampled_pair_branch_budget": float(total_branch_budget),
        "igsd/sampled_pair_selected_pair_count": float(total_selected_pairs),
        # Keep the logical pair count comparable across reference modes.  The
        # fixed-S mode additionally reports its deduplicated request/terminal
        # counts below so compute savings are explicit rather than hidden.
        "igsd/sampled_pair_planned_branch_count": float(2 * total_selected_pairs),
        "igsd/sampled_pair_budget_utilization": float(
            (2 * total_selected_pairs) / max(total_branch_budget, 1)
        ),
        "igsd/sampled_pair_eligible_count": float(total_eligible),
        "igsd/sampled_pair_budget_skipped_count": float(total_budget_skipped),
        "igsd/sampled_pair_log_odds_epsilon": float(log_odds_epsilon),
        "igsd/sampled_pair_log_odds_shift_mean": float(np.mean(all_log_odds_shifts))
        if all_log_odds_shifts
        else 0.0,
        "igsd/sampled_pair_log_odds_shift_std": float(np.std(all_log_odds_shifts))
        if all_log_odds_shifts
        else 0.0,
        "igsd/sampled_pair_log_odds_shift_positive_frac": float(
            np.mean(np.asarray(all_log_odds_shifts) > log_odds_epsilon)
        )
        if all_log_odds_shifts
        else 0.0,
        "igsd/sampled_pair_reference_mode_is_fixed_s": float(reference_mode == "fixed_s"),
        "igsd/sampled_pair_candidate_mode_is_student_top1": float(candidate_mode == "student_top1"),
        "igsd/sampled_pair_gate_source_is_environment_ig": float(
            gate_source == "environment_ig"
        ),
        "igsd/sampled_pair_gate_source_is_likelihood_gap": float(
            gate_source == "likelihood_gap"
        ),
        "igsd/sampled_pair_gate_source_is_constant": float(gate_source == "constant"),
        "igsd/sampled_pair_audit_all_eligible": float(audit_all_eligible),
        "igsd/sampled_pair_audit_selected_coverage": float(
            total_selected_pairs / max(total_eligible, 1)
        ),
        "igsd/sampled_pair_budget_selected_count": float(
            sum(
                bool(plan.get("budget_selected", False))
                for plans in plans_by_example
                for plan in plans
            )
        ),
        "igsd/sampled_pair_normal_pair_capacity": float(total_normal_pair_capacity),
        "igsd/sampled_pair_reference_terminal_count": float(
            sum(bool(plans) and any(plan.get("selected", False) for plan in plans) for plans in plans_by_example)
            if gate_source == "environment_ig" and reference_mode == "fixed_s"
            else 0.0
        ),
        "igsd/sampled_pair_generated_request_count": float(len(requests)),
        "igsd/sampled_pair_all_equal_count": float(identity_counts["all_equal"]),
        "igsd/sampled_pair_teacher_equals_student_top1_only_count": float(
            identity_counts["teacher_equals_student_top1_only"]
        ),
        "igsd/sampled_pair_teacher_equals_sampled_only_count": float(
            identity_counts["teacher_equals_sampled_only"]
        ),
        "igsd/sampled_pair_student_top1_equals_sampled_only_count": float(
            identity_counts["student_top1_equals_sampled_only"]
        ),
        "igsd/sampled_pair_all_distinct_count": float(identity_counts["all_distinct"]),
    }
    for name, values in diagnostic_values.items():
        metrics[f"igsd/sampled_pair_{name}_mean"] = float(np.mean(values)) if values else 0.0
    for state, count in prefilter_state_counts.items():
        metrics[f"igsd/sampled_pair_prefilter_state_{state}_count"] = float(count)
    for state, count in final_state_counts.items():
        metrics[f"igsd/sampled_pair_state_{state}_count"] = float(count)
        metrics[f"igsd/sampled_pair_final_state_{state}_count"] = float(count)
    return requests, records_by_example, plans_by_example, metrics


def build_sampled_action_pair_targets(
    examples: list[dict[str, Any]],
    *,
    utility_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack routed ``[teacher, reference]`` targets for compact pair-JSD.

    The tensors retain the complete sampled action width for position alignment,
    but ``active_mask`` is true only for a selected pair whose configured gate
    signal is valid and produces positive weight. Environment and likelihood
    signals must also exceed ``utility_margin``; constant controls bypass it.
    Inactive positions
    intentionally receive zero-filled IDs/log-probabilities; callers must use
    ``active_mask`` before materializing logits or invoking the pair helper.
    The reference defaults to the sampled action; modal-pair mode
    instead supplies the student top-1 token while retaining the sampled action
    as the on-policy training response.
    """

    if not math.isfinite(float(utility_margin)):
        raise ValueError(f"utility_margin must be finite, got {utility_margin}")
    if not examples:
        empty_ids = torch.empty((0, 0, 2), dtype=torch.long)
        empty_logs = torch.empty((0, 0, 2), dtype=torch.float32)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)
        return empty_ids, empty_logs, empty_mask

    response_lengths = [len(ex.get("student_action_ids", [])) for ex in examples]
    max_response = max(response_lengths, default=0)
    candidate_ids = torch.zeros((len(examples), max_response, 2), dtype=torch.long)
    teacher_log_probs = torch.zeros((len(examples), max_response, 2), dtype=torch.float32)
    active_mask = torch.zeros((len(examples), max_response), dtype=torch.bool)

    for row_idx, ex in enumerate(examples):
        action_ids = [int(value) for value in ex.get("student_action_ids", [])]
        query_mask = [bool(value) for value in ex.get("student_action_query_mask", [])]
        if len(action_ids) != len(query_mask):
            raise ValueError(
                "Sampled pair target action/query mask must align, got "
                f"{len(action_ids)=} and {len(query_mask)=}"
            )
        plans = ex.get("igsd_token_budget_plan", [])
        query_positions = [idx for idx, value in enumerate(query_mask) if value]
        if len(plans) != len(query_positions):
            raise ValueError(
                "Sampled pair target plans must align with sampled query positions, got "
                f"{len(plans)=} and {len(query_positions)=}"
            )
        gains = [float(value) for value in ex.get("igsd_token_gains", [])]
        gates = [float(value) for value in ex.get("igsd_token_gate", [])]
        gate_valid = [bool(value) for value in ex.get("igsd_token_gate_valid", [])]
        delta_valid = [bool(value) for value in ex.get("igsd_token_delta_valid", [])]
        expected = len(query_positions)
        if not (len(gains) == len(gates) == len(gate_valid) == len(delta_valid) == expected):
            raise ValueError(
                "Sampled pair token diagnostics must align with query positions, got "
                f"{len(gains)=}, {len(gates)=}, {len(gate_valid)=}, "
                f"{len(delta_valid)=}, {expected=}"
            )

        for token_idx, (plan, action_position) in enumerate(zip(plans, query_positions, strict=True)):
            gain = gains[token_idx]
            gate = gates[token_idx]
            gate_source = str(plan.get("pair_gate_source", "environment_ig")).lower()
            raw_gate_signal = plan.get("pair_gate_signal")
            gate_signal = gain if raw_gate_signal is None else float(raw_gate_signal)
            gate_signal_valid = bool(
                plan.get("pair_gate_valid", plan.get("pair_ig_valid", False))
            )
            # ``should_activate`` is intentionally stricter than merely being
            # selected: it describes a pair that is expected to contribute a
            # positive candidate-JSD weight.  Once a plan reaches this state,
            # malformed target metadata must fail closed with an explicit
            # error.  Silently skipping it would leave a positive token weight
            # with an all-zero compact target and make the run look valid while
            # dropping supervision.
            should_activate = (
                bool(plan.get("selected", False))
                and gate_signal_valid
                and gate_valid[token_idx]
                and delta_valid[token_idx]
                and math.isfinite(gain)
                and math.isfinite(gate_signal)
                and (
                    gate_source == "constant"
                    or gate_signal > float(utility_margin)
                )
                and math.isfinite(gate)
                and gate > 0.0
            )
            if not should_activate:
                continue
            teacher_id = int(plan.get("teacher_top1_id", -1))
            sampled_id = int(plan.get("sampled_token_id", -1))
            reference_id = int(plan.get("reference_token_id", sampled_id))
            teacher_log_prob = float(plan.get("teacher_top1_log_prob", float("nan")))
            reference_log_prob = float(
                plan.get(
                    "teacher_reference_log_prob",
                    plan.get("teacher_sample_log_prob", float("nan")),
                )
            )
            if teacher_id < 0 or reference_id < 0 or sampled_id < 0:
                raise ValueError(
                    "Sampled pair target has an invalid candidate token ID for a positive plan: "
                    f"row={row_idx}, position={action_position}, "
                    f"{teacher_id=}, {reference_id=}, {sampled_id=}"
                )
            if teacher_id == reference_id:
                raise ValueError(
                    "Sampled pair target has duplicate candidate IDs for a positive plan: "
                    f"row={row_idx}, position={action_position}, token_id={teacher_id}"
                )
            if action_ids[action_position] != sampled_id:
                raise ValueError(
                    "Sampled pair target is misaligned with the stored on-policy action at "
                    f"row={row_idx}, position={action_position}: "
                    f"response={action_ids[action_position]}, sampled={sampled_id}"
                )
            if not math.isfinite(teacher_log_prob) or not math.isfinite(reference_log_prob):
                raise ValueError(
                    "Sampled pair target has non-finite teacher candidate log-probabilities "
                    "for a positive plan: "
                    f"row={row_idx}, position={action_position}, "
                    f"{teacher_log_prob=}, {reference_log_prob=}"
                )
            if teacher_log_prob > 0.0 or reference_log_prob > 0.0:
                raise ValueError(
                    "Sampled pair target log-probabilities must be <= 0 for a positive plan: "
                    f"row={row_idx}, position={action_position}, "
                    f"{teacher_log_prob=}, {reference_log_prob=}"
                )
            candidate_ids[row_idx, action_position] = torch.tensor([teacher_id, reference_id])
            teacher_log_probs[row_idx, action_position] = torch.tensor(
                [teacher_log_prob, reference_log_prob]
            )
            active_mask[row_idx, action_position] = True

    return candidate_ids, teacher_log_probs, active_mask


def build_sampled_action_positive_mask(
    examples: list[dict[str, Any]],
    *,
    response_width: int | None = None,
    utility_margin: float = 0.0,
) -> torch.Tensor:
    """Pack gate-active sampled-pair positions for a top-k objective.

    The planner and scorer remain the source of truth for selection and gate
    validity.  This helper only projects the final positive pair state onto the
    padded sampled-action response axis so a top-k JSD can use exactly the same
    sparse supervision as candidate-pair JSD.
    """

    if not math.isfinite(float(utility_margin)):
        raise ValueError(f"utility_margin must be finite, got {utility_margin}")
    if not examples:
        width = max(int(response_width or 0), 0)
        return torch.empty((0, width), dtype=torch.bool)

    response_lengths = [len(ex.get("student_action_ids", [])) for ex in examples]
    inferred_width = max(response_lengths, default=0)
    width = inferred_width if response_width is None else int(response_width)
    if width < 0:
        raise ValueError(f"sampled-action positive mask response width must be non-negative, got {width}")
    if width < inferred_width:
        raise ValueError(
            "sampled-action positive mask response width is shorter than an action row, "
            f"got {width=} and {inferred_width=}"
        )
    active_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for row_idx, ex in enumerate(examples):
        action_ids = [int(value) for value in ex.get("student_action_ids", [])]
        query_mask = [bool(value) for value in ex.get("student_action_query_mask", [])]
        if len(action_ids) != len(query_mask):
            raise ValueError(
                "sampled-action positive mask action/query mask must align, got "
                f"{len(action_ids)=} and {len(query_mask)=}"
            )
        query_positions = [idx for idx, value in enumerate(query_mask) if value]
        plans = ex.get("igsd_token_budget_plan", [])
        gains = [float(value) for value in ex.get("igsd_token_gains", [])]
        gates = [float(value) for value in ex.get("igsd_token_gate", [])]
        gate_valid = [bool(value) for value in ex.get("igsd_token_gate_valid", [])]
        delta_valid = [bool(value) for value in ex.get("igsd_token_delta_valid", [])]
        expected = len(query_positions)
        if len(plans) != expected or not (
            len(gains) == len(gates) == len(gate_valid) == len(delta_valid) == expected
        ):
            raise ValueError(
                "sampled-action positive mask diagnostics must align with query positions, got "
                f"{len(plans)=}, {len(gains)=}, {len(gates)=}, {len(gate_valid)=}, "
                f"{len(delta_valid)=}, {expected=}"
            )
        for token_idx, action_position in enumerate(query_positions):
            plan = plans[token_idx]
            gate_source = str(plan.get("pair_gate_source", "environment_ig")).lower()
            raw_gate_signal = plan.get("pair_gate_signal")
            gate_signal = gains[token_idx] if raw_gate_signal is None else float(raw_gate_signal)
            gate_signal_valid = bool(
                plan.get("pair_gate_valid", plan.get("pair_ig_valid", False))
            )
            active = (
                bool(plan.get("selected", False))
                and gate_signal_valid
                and gate_valid[token_idx]
                and delta_valid[token_idx]
                and math.isfinite(gains[token_idx])
                and math.isfinite(gate_signal)
                and (gate_source == "constant" or gate_signal > float(utility_margin))
                and math.isfinite(gates[token_idx])
                and gates[token_idx] > 0.0
            )
            if active:
                active_mask[row_idx, action_position] = True
    return active_mask


def attach_token_intervention_outputs(
    records_by_example: list[list[dict[str, Any]]],
    outputs: list[dict[str, Any]],
    tokenizer: Any,
    *,
    retain_branch_action_ids: bool = False,
    max_queries_per_tool_call: int | None = None,
    prior_queries_by_example: list[list[str]] | None = None,
) -> dict[str, float]:
    """Parse raw exact-prefix outputs and mark malformed branches invalid."""

    branch_count = 0
    continuation_branch_count = 0
    request_count = 0
    error_count = 0
    valid_count = 0
    continuation_valid_count = 0
    metadata_mismatch_count = 0
    generation_limit_hit_count = 0
    generated_token_counts: list[int] = []
    completed_stop_count = 0
    aborted_stop_count = 0
    too_many_queries_count = 0
    duplicate_query_count = 0
    for records in records_by_example:
        for record in records:
            branch_count += 1
            continuation_branch_count += int(
                record.get("branch_kind")
                in {
                    "teacher_continuation",
                    "budgeted_teacher_top1",
                    "budgeted_teacher_top2",
                    "sampled_pair_teacher",
                    "sampled_pair_student",
                }
            )
            request_index = record.get("request_index")
            if request_index is None:
                if record.get("branch_kind") == "student_terminal" and record.get("branch_valid"):
                    valid_count += 1
                continue
            request_count += 1
            result = outputs[request_index] if request_index < len(outputs) else None
            if not result or result.get("error") or result.get("output") is None:
                record["branch_error"] = (result or {}).get("error", "missing_rollout_output")
                error_count += 1
                continue
            result_example_idx = result.get("example_index")
            result_branch_idx = result.get("branch_index")
            if (
                result_example_idx is not None
                and int(result_example_idx) != int(record["example_index"])
            ) or (
                result_branch_idx is not None
                and int(result_branch_idx) != int(record["branch_index"])
            ):
                record["branch_error"] = "rollout_output_metadata_mismatch"
                metadata_mismatch_count += 1
                error_count += 1
                continue
            output = result["output"]
            if isinstance(output, dict):
                generated = output.get("token_ids", [])
                stop_reason = output.get("stop_reason")
            else:
                generated = getattr(output, "token_ids", [])
                stop_reason = getattr(output, "stop_reason", None)
            generated_ids = [int(token_id) for token_id in generated]
            record["generated_token_count"] = len(generated_ids)
            record["stop_reason"] = stop_reason
            normalized_stop_reason = str(stop_reason).strip().lower()
            max_new_tokens = record.get("max_new_tokens")
            hit_generation_limit = bool(
                max_new_tokens is not None and len(generated_ids) >= int(max_new_tokens)
            )
            record["hit_generation_limit"] = hit_generation_limit
            generation_limit_hit_count += int(hit_generation_limit)
            generated_token_counts.append(len(generated_ids))
            completed_stop_count += int(
                normalized_stop_reason in {"completed", "length", "stop"}
            )
            aborted_stop_count += int(normalized_stop_reason in {"abort", "aborted"})
            if normalized_stop_reason in {"abort", "aborted"}:
                record["branch_error"] = "rollout_aborted"
                error_count += 1
                continue
            expected_first_generated_token = record.get("expected_first_generated_token")
            if (
                expected_first_generated_token is not None
                and (not generated_ids or generated_ids[0] != int(expected_first_generated_token))
            ):
                record["branch_error"] = "teacher_top1_generation_mismatch"
                error_count += 1
                continue
            action_ids = list(record.get("fixed_action_ids", [])) + generated_ids
            if retain_branch_action_ids:
                # P0 terminal diagnostics resume from the exact action emitted
                # by this branch. Ordinary training leaves this disabled to
                # avoid retaining up to max_new_tokens Python integers per
                # branch after the query has already been parsed.
                record["generated_token_ids"] = generated_ids
                record["branch_action_ids"] = action_ids
            action_text = tokenizer.decode(action_ids, skip_special_tokens=False)
            queries = extract_search_queries(action_text, strict_schema=True)
            if not queries:
                record["branch_error"] = "malformed_or_truncated_tool_call"
                error_count += 1
                continue
            if (
                max_queries_per_tool_call is not None
                and int(max_queries_per_tool_call) > 0
                and len(queries) > int(max_queries_per_tool_call)
            ):
                record["branch_error"] = "too_many_queries"
                record["branch_queries"] = queries
                record["branch_query"] = " ".join(queries)
                too_many_queries_count += 1
                error_count += 1
                continue
            prior_queries = (
                prior_queries_by_example[int(record["example_index"])]
                if prior_queries_by_example is not None
                and 0 <= int(record["example_index"]) < len(prior_queries_by_example)
                else []
            )
            seen_queries = {
                normalized
                for query in prior_queries
                if (normalized := _normalize_search_query_for_dedup(query))
            }
            has_duplicate = False
            for query in queries:
                normalized = _normalize_search_query_for_dedup(query)
                if not normalized or normalized in seen_queries:
                    has_duplicate = True
                    break
                seen_queries.add(normalized)
            if has_duplicate:
                record["branch_error"] = "duplicate_query"
                record["branch_queries"] = queries
                record["branch_query"] = " ".join(queries)
                duplicate_query_count += 1
                error_count += 1
                continue
            record["branch_queries"] = queries
            record["branch_query"] = " ".join(queries)
            record["branch_valid"] = True
            valid_count += 1
            continuation_valid_count += 1

    return {
        "igsd/token_intervention_branch_count": float(branch_count),
        "igsd/token_intervention_continuation_branch_count": float(continuation_branch_count),
        "igsd/token_intervention_request_count": float(request_count),
        "igsd/token_intervention_request_eligible_frac": float(
            request_count / max(continuation_branch_count, 1)
        ),
        "igsd/token_intervention_received_output_count": float(len(outputs)),
        "igsd/token_intervention_output_count_mismatch": float(len(outputs) != request_count),
        "igsd/token_intervention_output_metadata_mismatch_count": float(metadata_mismatch_count),
        "igsd/token_intervention_generation_limit_hit_count": float(generation_limit_hit_count),
        "igsd/token_intervention_generation_limit_hit_frac": float(
            generation_limit_hit_count / max(request_count, 1)
        ),
        "igsd/token_intervention_generated_token_mean": float(
            np.mean(generated_token_counts) if generated_token_counts else 0.0
        ),
        "igsd/token_intervention_generated_token_max": float(
            max(generated_token_counts, default=0)
        ),
        "igsd/token_intervention_completed_stop_frac": float(
            completed_stop_count / max(len(generated_token_counts), 1)
        ),
        "igsd/token_intervention_aborted_stop_frac": float(
            aborted_stop_count / max(len(generated_token_counts), 1)
        ),
        "igsd/token_intervention_rollout_error_count": float(error_count),
        "igsd/token_intervention_too_many_queries_count": float(too_many_queries_count),
        "igsd/token_intervention_duplicate_query_count": float(duplicate_query_count),
        "igsd/token_intervention_continuation_parse_valid_count": float(
            continuation_valid_count
        ),
        "igsd/token_intervention_continuation_parse_valid_frac": float(
            continuation_valid_count / max(request_count, 1)
        ),
        "igsd/token_intervention_branch_parse_valid_frac": float(valid_count / max(branch_count, 1)),
    }


def _token_gate_and_row_normalize(
    gains: np.ndarray,
    valid_mask: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map signed branch gains to non-negative token-level loss weights."""

    mode = str(config.get("igsd_token_gate_mode", "sigmoid")).lower()
    beta = float(config.get("igsd_token_gate_beta", 5.0))
    margin = float(config.get("igsd_token_gate_margin", 0.0))
    normalization = str(config.get("igsd_token_gate_normalization", "row_mean")).lower()
    invalid_fallback_weight = float(
        config.get("igsd_token_invalid_fallback_weight", 0.0)
    )
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"algorithm.igsd_token_gate_beta must be finite and positive, got {beta}")
    if not np.isfinite(margin):
        raise ValueError(f"algorithm.igsd_token_gate_margin must be finite, got {margin}")
    if (
        not np.isfinite(invalid_fallback_weight)
        or invalid_fallback_weight < 0.0
        or invalid_fallback_weight > 1.0
    ):
        raise ValueError(
            "algorithm.igsd_token_invalid_fallback_weight must be finite and in [0, 1], "
            f"got {invalid_fallback_weight}"
        )
    valid_mask = np.asarray(valid_mask, dtype=bool)
    gains = np.asarray(gains, dtype=np.float64)
    if gains.shape != valid_mask.shape:
        raise ValueError(
            "token gains and valid mask must have the same shape, "
            f"got {gains.shape} vs {valid_mask.shape}"
        )
    finite_mask = np.isfinite(gains)
    valid_mask = valid_mask & finite_mask
    raw = np.zeros_like(gains, dtype=np.float64)
    if mode == "sigmoid":
        raw[valid_mask] = 1.0 / (1.0 + np.exp(-np.clip(beta * (gains[valid_mask] - margin), -60.0, 60.0)))
    elif mode == "rectified_sigmoid":
        sigmoid = 1.0 / (
            1.0
            + np.exp(-np.clip(beta * (gains[valid_mask] - margin), -60.0, 60.0))
        )
        raw[valid_mask] = np.maximum(0.0, 2.0 * sigmoid - 1.0)
    elif mode == "hard":
        raw[valid_mask] = (gains[valid_mask] > margin).astype(np.float64)
    elif mode == "none":
        raw[valid_mask] = 1.0
    else:
        raise ValueError(
            "Unsupported algorithm.igsd_token_gate_mode: "
            f"{mode!r}; expected one of ['sigmoid', 'rectified_sigmoid', 'hard', 'none']"
        )

    valid_count = int(valid_mask.sum())
    gate_mean = float(raw[valid_mask].mean()) if valid_count else 0.0
    if normalization == "row_mean" and gate_mean > 0.0:
        normalized = np.zeros_like(raw)
        normalized[valid_mask] = raw[valid_mask] / gate_mean
    elif normalization == "row_mean":
        normalized = np.zeros_like(raw)
    elif normalization == "none":
        normalized = raw.copy()
    else:
        raise ValueError(
            "Unsupported algorithm.igsd_token_gate_normalization: "
            f"{normalization!r}; expected one of ['row_mean', 'none']"
        )
    fallback_mask = ~valid_mask if invalid_fallback_weight > 0.0 else np.zeros_like(valid_mask)
    normalized[fallback_mask] = invalid_fallback_weight
    effective_mask = valid_mask | fallback_mask
    effective_count = int(effective_mask.sum())
    output_gate_mean = float(normalized[effective_mask].mean()) if effective_count else 0.0
    metrics = {
        "igsd/token_intervention_valid_token_count": float(valid_count),
        "igsd/token_intervention_valid_token_frac": float(valid_count / max(len(valid_mask), 1)),
        "igsd/token_intervention_raw_gate_mean": gate_mean,
        "igsd/token_intervention_raw_gate_max": (
            float(raw[valid_mask].max()) if valid_count else 0.0
        ),
        "igsd/token_intervention_normalized_gate_mean": output_gate_mean,
        "igsd/token_intervention_normalized_gate_max": (
            float(normalized[effective_mask].max()) if effective_count else 0.0
        ),
        "igsd/token_intervention_output_gate_mean": output_gate_mean,
        "igsd/token_intervention_output_gate_max": (
            float(normalized[effective_mask].max()) if effective_count else 0.0
        ),
        "igsd/token_intervention_effective_token_count": float(effective_count),
        "igsd/token_intervention_fallback_token_count": float(fallback_mask.sum()),
        "igsd/token_intervention_fallback_token_frac": float(
            fallback_mask.sum() / max(len(valid_mask), 1)
        ),
        "igsd/token_intervention_gate_normalization_is_row_mean": float(
            normalization == "row_mean"
        ),
        "igsd/token_intervention_gate_normalization_is_none": float(
            normalization == "none"
        ),
        "igsd/token_intervention_positive_gate_frac": (
            float(np.mean(normalized[effective_mask] > 0.0)) if effective_count else 0.0
        ),
        "igsd/token_intervention_gate_zero_frac": (
            float(np.mean(normalized[effective_mask] <= 0.0)) if effective_count else 0.0
        ),
    }
    return normalized.astype(np.float32), metrics


def _sampled_pair_gate_from_gains(
    gains: np.ndarray,
    valid_mask: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float]:
    """Build token- or query-granularity weights from sampled-pair Delta-IG.

    ``query_mean`` averages only routed pairs with a finite environment score,
    preserves the sign of that mean, and applies one shared gate to those same
    pairs. It never expands supervision to an unrouted or invalid position.
    The per-token ``gains`` remain available separately for diagnostics.
    """

    gains = np.asarray(gains, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if gains.shape != valid_mask.shape:
        raise ValueError(
            "sampled-pair gains and valid mask must have the same shape, "
            f"got {gains.shape} vs {valid_mask.shape}"
        )
    valid_mask = valid_mask & np.isfinite(gains)
    granularity = str(
        config.get("igsd_sampled_pair_gate_granularity", "token")
    ).lower()
    if granularity not in {"token", "query_mean"}:
        raise ValueError(
            "algorithm.igsd_sampled_pair_gate_granularity must be 'token' or "
            f"'query_mean', got {granularity!r}"
        )
    margin = float(config.get("igsd_token_gate_margin", 0.0))
    gate_signals = np.zeros_like(gains, dtype=np.float64)
    query_mean_gain = float(gains[valid_mask].mean()) if valid_mask.any() else 0.0
    query_shared_gate = 0.0

    if granularity == "token":
        gate_signals[valid_mask] = gains[valid_mask]
        gates, metrics = _token_gate_and_row_normalize(gains, valid_mask, config)
    else:
        scalar_signal = np.asarray([query_mean_gain], dtype=np.float64)
        scalar_valid = np.asarray([bool(valid_mask.any())], dtype=bool)
        scalar_gates, metrics = _token_gate_and_row_normalize(
            scalar_signal,
            scalar_valid,
            config,
        )
        query_shared_gate = float(scalar_gates[0]) if scalar_valid[0] else 0.0
        gate_signals[valid_mask] = query_mean_gain
        gates = np.zeros_like(gains, dtype=np.float32)
        gates[valid_mask] = query_shared_gate

    gates[~(valid_mask & (gate_signals > margin))] = 0.0
    if granularity == "query_mean":
        query_shared_gate = float(gates[valid_mask][0]) if valid_mask.any() else 0.0
    metrics["igsd/sampled_pair_gate_granularity_is_token"] = float(
        granularity == "token"
    )
    metrics["igsd/sampled_pair_gate_granularity_is_query_mean"] = float(
        granularity == "query_mean"
    )
    metrics["igsd/sampled_pair_query_mean_gain"] = query_mean_gain
    metrics["igsd/sampled_pair_query_shared_gate"] = query_shared_gate
    return (
        gates.astype(np.float32),
        gate_signals.astype(np.float32),
        metrics,
        query_mean_gain,
        query_shared_gate,
    )


def select_search_turn_indices(num_turns: int, mode: str, limit: int) -> list[int]:
    """Select search-turn indices without changing the legacy last-turn path."""

    if num_turns <= 0:
        return []
    if mode == "last_search":
        return [num_turns - 1]
    if mode != "all_search":
        raise ValueError(f"Unsupported IGSD teacher turn selection: {mode!r}")
    if limit <= 0 or limit >= num_turns:
        return list(range(num_turns))
    if limit == 1:
        return [num_turns - 1]

    # Spread a fixed budget across the full trajectory instead of silently
    # turning all_search into last-k search turns.
    selected = {
        int(round(slot * (num_turns - 1) / (limit - 1)))
        for slot in range(limit)
    }
    return sorted(selected)


def _turn_bucket(turn_idx: int, turn_count: int) -> str:
    if turn_count <= 1:
        return "only"
    if turn_idx <= 0:
        return "first"
    if turn_idx >= turn_count - 1:
        return "last"
    return "middle"


def apply_turn_routing(
    examples: list[dict[str, Any]],
    mode: str,
    gate_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Route distillation weight across turns from the same failed trajectory.

    ``ig_gate`` remains the raw paired-IG gate for diagnostics. The routed
    ``ig_turn_weight`` is the value consumed by the distillation loss.
    """

    supported = {"independent", "prompt_normalized", "top1_positive"}
    if mode not in supported:
        raise ValueError(f"Unsupported IGSD turn routing mode {mode!r}; expected one of {sorted(supported)}")
    if not examples:
        metrics = {
            "igsd/m2_turn_routing_is_independent": float(mode == "independent"),
            "igsd/m2_turn_routing_is_prompt_normalized": float(mode == "prompt_normalized"),
            "igsd/m2_turn_routing_is_top1_positive": float(mode == "top1_positive"),
            "igsd/m2_valid_turns_per_prompt_mean": 0.0,
            "igsd/m2_multi_valid_turn_prompt_frac": 0.0,
            "igsd/m2_prompts_with_multiple_positive_turns_frac": 0.0,
            "igsd/m2_random_turn_top1_baseline": 0.0,
            "igsd/m2_random_last_turn_baseline": 0.0,
            "igsd/m2_max_delta_turn_is_last_frac": 0.0,
            "igsd/m2_max_delta_turn_is_last_lift_over_random": 0.0,
            "igsd/m2_max_delta_turn_idx_mean": 0.0,
            "igsd/m2_max_delta_relative_turn_mean": 0.0,
            "igsd/m2_max_delta_mean": 0.0,
            "igsd/m2_max_delta_positive_frac": 0.0,
            "igsd/m2_routed_turn_count": 0.0,
            "igsd/m2_routed_prompt_count": 0.0,
            "igsd/m2_routed_turn_frac": 0.0,
            "igsd/m2_routed_turns_per_prompt_mean": 0.0,
            "igsd/m2_routed_turn_is_last_frac": 0.0,
            "igsd/m2_routed_relative_turn_mean": 0.0,
            "igsd/m2_routed_weight_mean": 0.0,
            "igsd/m2_routed_weight_sum": 0.0,
            "igsd/m2_routed_weight_max": 0.0,
        }
        for bucket in ("only", "first", "middle", "last"):
            metrics[f"igsd/m2_max_delta_bucket_{bucket}_frac"] = 0.0
        return examples, metrics

    groups: dict[int, list[int]] = {}
    for idx, ex in enumerate(examples):
        groups.setdefault(int(ex["source_sample_idx"]), []).append(idx)

    routed_weights = np.zeros(len(examples), dtype=np.float32)
    active_counts: list[int] = []
    if mode == "prompt_normalized":
        for indices in groups.values():
            active = [idx for idx in indices if float(examples[idx]["ig_gate"]) > 0.0]
            if active:
                active_counts.append(len(active))
        global_mean_active = float(np.mean(active_counts)) if active_counts else 0.0
    else:
        global_mean_active = 0.0

    for indices in groups.values():
        if mode == "independent":
            for idx in indices:
                routed_weights[idx] = float(examples[idx]["ig_gate"])
            continue

        if mode == "top1_positive":
            eligible = [
                idx
                for idx in indices
                if float(examples[idx].get("ig_gate_signal", examples[idx]["ig_delta"])) > gate_margin
            ]
            if eligible:
                best_idx = max(
                    eligible,
                    key=lambda idx: float(
                        examples[idx].get("ig_gate_signal", examples[idx]["ig_delta"])
                    ),
                )
                routed_weights[best_idx] = float(examples[best_idx]["ig_gate"])
            continue

        active = [idx for idx in indices if float(examples[idx]["ig_gate"]) > 0.0]
        if not active:
            continue
        gate_sum = sum(float(examples[idx]["ig_gate"]) for idx in active)
        prompt_confidence = max(float(examples[idx]["ig_gate"]) for idx in active)
        if gate_sum <= 0.0:
            continue
        for idx in active:
            routed_weights[idx] = (
                global_mean_active
                * float(examples[idx]["ig_gate"])
                / gate_sum
                * prompt_confidence
            )

    for idx, ex in enumerate(examples):
        ex["ig_turn_weight"] = float(routed_weights[idx])
        ex["ig_turn_selected"] = bool(routed_weights[idx] > 0.0)
        turn_count = max(int(ex.get("source_turn_count", 1)), 1)
        turn_idx = int(ex.get("source_turn_idx", 0))
        ex["source_relative_turn"] = float(turn_idx / max(turn_count - 1, 1))
        ex["source_turn_bucket"] = _turn_bucket(turn_idx, turn_count)

    group_values = list(groups.values())
    positive_counts = [
        sum(float(examples[idx]["ig_delta"]) > 0.0 for idx in indices)
        for indices in group_values
    ]
    routed_counts = [
        sum(float(routed_weights[idx]) > 0.0 for idx in indices)
        for indices in group_values
    ]
    max_delta_examples: list[dict[str, Any]] = []
    for indices in group_values:
        best_idx = max(indices, key=lambda idx: float(examples[idx]["ig_delta"]))
        best = examples[best_idx]
        best["is_max_delta_turn"] = True
        max_delta_examples.append(best)
        for idx in indices:
            examples[idx].setdefault("is_max_delta_turn", False)
    routed_examples = [examples[idx] for idx, weight in enumerate(routed_weights) if float(weight) > 0.0]
    random_top1_baseline = float(np.mean([1.0 / len(indices) for indices in group_values]))
    random_last_baseline = float(
        np.mean(
            [
                sum(
                    int(examples[idx].get("source_turn_idx", 0))
                    >= max(int(examples[idx].get("source_turn_count", 1)) - 1, 0)
                    for idx in indices
                )
                / len(indices)
                for indices in group_values
            ]
        )
    )
    max_delta_is_last_frac = float(
        np.mean(
            [
                int(ex.get("source_turn_idx", 0)) >= max(int(ex.get("source_turn_count", 1)) - 1, 0)
                for ex in max_delta_examples
            ]
        )
    )

    metrics = {
        "igsd/m2_turn_routing_is_independent": float(mode == "independent"),
        "igsd/m2_turn_routing_is_prompt_normalized": float(mode == "prompt_normalized"),
        "igsd/m2_turn_routing_is_top1_positive": float(mode == "top1_positive"),
        "igsd/m2_valid_turns_per_prompt_mean": float(np.mean([len(v) for v in group_values])),
        "igsd/m2_multi_valid_turn_prompt_frac": float(np.mean([len(v) > 1 for v in group_values])),
        "igsd/m2_prompts_with_multiple_positive_turns_frac": float(np.mean([count > 1 for count in positive_counts])),
        "igsd/m2_random_turn_top1_baseline": random_top1_baseline,
        "igsd/m2_random_last_turn_baseline": random_last_baseline,
        "igsd/m2_max_delta_turn_is_last_frac": max_delta_is_last_frac,
        "igsd/m2_max_delta_turn_is_last_lift_over_random": max_delta_is_last_frac - random_last_baseline,
        "igsd/m2_max_delta_turn_idx_mean": float(
            np.mean([int(ex.get("source_turn_idx", 0)) for ex in max_delta_examples])
        ),
        "igsd/m2_max_delta_relative_turn_mean": float(
            np.mean([float(ex["source_relative_turn"]) for ex in max_delta_examples])
        ),
        "igsd/m2_max_delta_mean": float(np.mean([float(ex["ig_delta"]) for ex in max_delta_examples])),
        "igsd/m2_max_delta_positive_frac": float(
            np.mean([float(ex["ig_delta"]) > 0.0 for ex in max_delta_examples])
        ),
        "igsd/m2_routed_turn_count": float(np.count_nonzero(routed_weights > 0.0)),
        "igsd/m2_routed_prompt_count": float(sum(count > 0 for count in routed_counts)),
        "igsd/m2_routed_turn_frac": float(np.mean(routed_weights > 0.0)),
        "igsd/m2_routed_turns_per_prompt_mean": float(np.mean(routed_counts)),
        "igsd/m2_routed_turn_is_last_frac": float(
            np.mean(
                [
                    int(ex.get("source_turn_idx", 0)) >= max(int(ex.get("source_turn_count", 1)) - 1, 0)
                    for ex in routed_examples
                ]
            )
            if routed_examples
            else 0.0
        ),
        "igsd/m2_routed_relative_turn_mean": float(
            np.mean([float(ex["source_relative_turn"]) for ex in routed_examples])
            if routed_examples
            else 0.0
        ),
        "igsd/m2_routed_weight_mean": float(routed_weights.mean()),
        "igsd/m2_routed_weight_sum": float(routed_weights.sum()),
        "igsd/m2_routed_weight_max": float(routed_weights.max()),
    }
    max_turn_indices = [int(ex.get("source_turn_idx", 0)) for ex in max_delta_examples]
    for turn_idx in sorted({int(ex.get("source_turn_idx", 0)) for ex in examples}):
        metrics[f"igsd/m2_max_delta_turn_{turn_idx}_frac"] = float(
            np.mean([value == turn_idx for value in max_turn_indices])
        )
    max_turn_buckets = [str(ex["source_turn_bucket"]) for ex in max_delta_examples]
    for bucket in ("only", "first", "middle", "last"):
        metrics[f"igsd/m2_max_delta_bucket_{bucket}_frac"] = float(
            np.mean([value == bucket for value in max_turn_buckets])
        )
    return examples, metrics


def turn_level_ig_metrics(examples: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize paired IG, gates, and routed weights by absolute/relative turn."""

    metrics: dict[str, float] = {}

    def add_group(prefix: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        metrics[f"{prefix}_valid_pair_count"] = float(len(rows))
        metrics[f"{prefix}_ig_delta_mean"] = float(np.mean([float(row["ig_delta"]) for row in rows]))
        metrics[f"{prefix}_positive_delta_frac"] = float(
            np.mean([float(row["ig_delta"]) > 0.0 for row in rows])
        )
        metrics[f"{prefix}_gate_mean"] = float(np.mean([float(row["ig_gate"]) for row in rows]))
        metrics[f"{prefix}_routed_weight_mean"] = float(
            np.mean([float(row.get("ig_turn_weight", row["ig_gate"])) for row in rows])
        )
        metrics[f"{prefix}_routed_selected_frac"] = float(
            np.mean([float(row.get("ig_turn_weight", row["ig_gate"])) > 0.0 for row in rows])
        )
        metrics[f"{prefix}_teacher_query_changed_frac"] = float(
            np.mean([row["teacher_query"].strip() != row["student_query"].strip() for row in rows])
        )

    turn_indices = sorted({int(ex.get("source_turn_idx", 0)) for ex in examples})
    for turn_idx in turn_indices:
        add_group(
            f"igsd/m2_turn_{turn_idx}",
            [ex for ex in examples if int(ex.get("source_turn_idx", 0)) == turn_idx],
        )
    for bucket in ("only", "first", "middle", "last"):
        add_group(
            f"igsd/m2_bucket_{bucket}",
            [ex for ex in examples if ex.get("source_turn_bucket") == bucket],
        )
    return metrics


def entropy_delta_metrics(examples: list[dict[str, Any]]) -> dict[str, float]:
    """Measure whether rollout entropy identifies high-delta turns beyond position bias."""

    def pearson(left: list[float], right: list[float]) -> float:
        if len(left) < 2 or len(left) != len(right):
            return 0.0
        left_arr = np.asarray(left, dtype=np.float64)
        right_arr = np.asarray(right, dtype=np.float64)
        if float(left_arr.std()) == 0.0 or float(right_arr.std()) == 0.0:
            return 0.0
        return float(np.corrcoef(left_arr, right_arr)[0, 1])

    def average_ranks(values: list[float]) -> list[float]:
        arr = np.asarray(values, dtype=np.float64)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(len(arr), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and arr[order[end]] == arr[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks.tolist()

    def weighted_mean(values: list[float], weights: list[float]) -> float:
        if not values or len(values) != len(weights):
            return 0.0
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return 0.0
        return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights)))

    def centered_pairs(groups: list[list[dict[str, Any]]], field: str) -> tuple[list[float], list[float]]:
        centered_entropy: list[float] = []
        centered_delta: list[float] = []
        for group in groups:
            group_entropy = np.asarray([float(ex[field]) for ex in group], dtype=np.float64)
            group_delta = np.asarray([float(ex["ig_delta"]) for ex in group], dtype=np.float64)
            centered_entropy.extend((group_entropy - group_entropy.mean()).tolist())
            centered_delta.extend((group_delta - group_delta.mean()).tolist())
        return centered_entropy, centered_delta

    def centered_rank_pairs(
        groups: list[list[dict[str, Any]]], field: str
    ) -> tuple[list[float], list[float]]:
        centered_entropy: list[float] = []
        centered_delta: list[float] = []
        for group in groups:
            group_entropy = np.asarray(
                average_ranks([float(ex[field]) for ex in group]), dtype=np.float64
            )
            group_delta = np.asarray(
                average_ranks([float(ex["ig_delta"]) for ex in group]), dtype=np.float64
            )
            centered_entropy.extend((group_entropy - group_entropy.mean()).tolist())
            centered_delta.extend((group_delta - group_delta.mean()).tolist())
        return centered_entropy, centered_delta

    def residualize(values: list[float], features: list[list[float]]) -> list[float]:
        if len(values) < 3 or len(values) != len(features):
            return []
        feature_arr = np.asarray(features, dtype=np.float64)
        if feature_arr.ndim != 2 or not np.all(np.isfinite(feature_arr)):
            return []
        standardized = np.zeros_like(feature_arr)
        for col_idx in range(feature_arr.shape[1]):
            column = feature_arr[:, col_idx]
            std = float(column.std())
            if std > 0.0:
                standardized[:, col_idx] = (column - column.mean()) / std
        design = np.column_stack([np.ones(len(values), dtype=np.float64), standardized])
        target = np.asarray(values, dtype=np.float64)
        fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        return (target - fitted).tolist()

    def is_last(row: dict[str, Any]) -> bool:
        turn_count = max(int(row.get("source_turn_count", 1)), 1)
        return int(row.get("source_turn_idx", 0)) >= turn_count - 1

    def add_signal(prefix: str, field: str, token_count_field: str, metrics: dict[str, float]) -> None:
        rows = [
            ex
            for ex in examples
            if ex.get(field) is not None
            and np.isfinite(float(ex[field]))
            and np.isfinite(float(ex["ig_delta"]))
        ]
        entropy_values = [float(ex[field]) for ex in rows]
        delta_values = [float(ex["ig_delta"]) for ex in rows]
        metrics[f"{prefix}_pair_count"] = float(len(rows))
        metrics[f"{prefix}_pair_coverage"] = float(len(rows) / max(len(examples), 1))
        metrics[f"{prefix}_delta_pearson"] = pearson(entropy_values, delta_values)
        metrics[f"{prefix}_delta_spearman"] = pearson(
            average_ranks(entropy_values), average_ranks(delta_values)
        )

        nonlast_rows = [row for row in rows if not is_last(row)]
        nonlast_entropy = [float(row[field]) for row in nonlast_rows]
        nonlast_delta = [float(row["ig_delta"]) for row in nonlast_rows]
        metrics[f"{prefix}_nonlast_pair_count"] = float(len(nonlast_rows))
        metrics[f"{prefix}_nonlast_pair_coverage"] = float(len(nonlast_rows) / max(len(rows), 1))
        metrics[f"{prefix}_nonlast_delta_pearson"] = pearson(nonlast_entropy, nonlast_delta)
        metrics[f"{prefix}_nonlast_delta_spearman"] = pearson(
            average_ranks(nonlast_entropy), average_ranks(nonlast_delta)
        )

        groups: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(int(row["source_sample_idx"]), []).append(row)
        multi_groups = [group for group in groups.values() if len(group) >= 2]

        centered_entropy, centered_delta = centered_pairs(multi_groups, field)
        centered_entropy_rank, centered_delta_rank = centered_rank_pairs(multi_groups, field)
        nonlast_groups = [
            group
            for group in ([row for row in group if not is_last(row)] for group in groups.values())
            if len(group) >= 2
        ]
        centered_nonlast_entropy, centered_nonlast_delta = centered_pairs(nonlast_groups, field)
        centered_nonlast_entropy_rank, centered_nonlast_delta_rank = centered_rank_pairs(
            nonlast_groups, field
        )
        top1_matches: list[float] = []
        top1_positive: list[float] = []
        top1_deltas: list[float] = []
        max_delta_entropy_ranks: list[float] = []
        for group in multi_groups:
            group_entropy = np.asarray([float(ex[field]) for ex in group], dtype=np.float64)
            group_delta = np.asarray([float(ex["ig_delta"]) for ex in group], dtype=np.float64)

            entropy_best = int(np.argmax(group_entropy))
            delta_best = int(np.argmax(group_delta))
            top1_matches.append(float(entropy_best == delta_best))
            top1_positive.append(float(group_delta[entropy_best] > 0.0))
            top1_deltas.append(float(group_delta[entropy_best]))
            descending_entropy = np.argsort(-group_entropy, kind="mergesort")
            delta_best_rank = int(np.where(descending_entropy == delta_best)[0][0])
            max_delta_entropy_ranks.append(delta_best_rank / max(len(group) - 1, 1))

        metrics[f"{prefix}_multi_turn_prompt_count"] = float(len(multi_groups))
        random_top1_baseline = (
            float(np.mean([1.0 / len(group) for group in multi_groups])) if multi_groups else 0.0
        )
        top1_match_frac = float(np.mean(top1_matches)) if top1_matches else 0.0
        top1_positive_frac = float(np.mean(top1_positive)) if top1_positive else 0.0
        all_positive_frac = float(np.mean([delta > 0.0 for delta in delta_values])) if delta_values else 0.0
        all_delta_mean = float(np.mean(delta_values)) if delta_values else 0.0
        top1_delta_mean = float(np.mean(top1_deltas)) if top1_deltas else 0.0
        metrics[f"{prefix}_within_prompt_delta_pearson"] = pearson(centered_entropy, centered_delta)
        metrics[f"{prefix}_within_prompt_delta_spearman"] = pearson(
            centered_entropy_rank, centered_delta_rank
        )
        metrics[f"{prefix}_nonlast_multi_turn_prompt_count"] = float(len(nonlast_groups))
        metrics[f"{prefix}_nonlast_within_prompt_delta_pearson"] = pearson(
            centered_nonlast_entropy, centered_nonlast_delta
        )
        metrics[f"{prefix}_nonlast_within_prompt_delta_spearman"] = pearson(
            centered_nonlast_entropy_rank, centered_nonlast_delta_rank
        )
        metrics[f"{prefix}_top1_random_match_baseline"] = random_top1_baseline
        metrics[f"{prefix}_top1_max_delta_match_frac"] = top1_match_frac
        metrics[f"{prefix}_top1_max_delta_match_lift"] = top1_match_frac - random_top1_baseline
        metrics[f"{prefix}_top1_positive_delta_frac"] = top1_positive_frac
        metrics[f"{prefix}_top1_positive_delta_lift"] = top1_positive_frac - all_positive_frac
        metrics[f"{prefix}_top1_delta_mean"] = top1_delta_mean
        metrics[f"{prefix}_top1_delta_mean_lift"] = top1_delta_mean - all_delta_mean
        # 0 means entropy ranks the true max-delta turn first; 1 means last.
        metrics[f"{prefix}_max_delta_normalized_rank_mean"] = (
            float(np.mean(max_delta_entropy_ranks)) if max_delta_entropy_ranks else 0.0
        )

        # Correlate only examples with the same absolute position and trajectory
        # length. This removes the dominant late-turn trend without pooling
        # incomparable first/middle/last states.
        position_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in rows:
            key = (int(row.get("source_turn_count", 1)), int(row.get("source_turn_idx", 0)))
            position_groups.setdefault(key, []).append(row)
        position_multi_groups = [group for group in position_groups.values() if len(group) >= 2]
        position_centered_entropy, position_centered_delta = centered_pairs(
            position_multi_groups, field
        )
        position_centered_entropy_rank, position_centered_delta_rank = centered_rank_pairs(
            position_multi_groups, field
        )
        position_pearsons: list[float] = []
        position_spearmans: list[float] = []
        position_weights: list[float] = []
        position_pair_count = 0
        for group in position_groups.values():
            group_entropy = [float(row[field]) for row in group]
            group_delta = [float(row["ig_delta"]) for row in group]
            if (
                len(group) < 3
                or float(np.std(group_entropy)) == 0.0
                or float(np.std(group_delta)) == 0.0
            ):
                continue
            position_pearsons.append(pearson(group_entropy, group_delta))
            position_spearmans.append(
                pearson(average_ranks(group_entropy), average_ranks(group_delta))
            )
            position_weights.append(float(len(group)))
            position_pair_count += len(group)
        metrics[f"{prefix}_position_stratum_count"] = float(len(position_pearsons))
        metrics[f"{prefix}_position_conditioned_pair_count"] = float(position_pair_count)
        metrics[f"{prefix}_position_fixed_effect_pair_count"] = float(
            len(position_centered_entropy)
        )
        metrics[f"{prefix}_position_fixed_effect_delta_pearson"] = pearson(
            position_centered_entropy, position_centered_delta
        )
        metrics[f"{prefix}_position_fixed_effect_delta_spearman"] = pearson(
            position_centered_entropy_rank, position_centered_delta_rank
        )
        metrics[f"{prefix}_position_conditioned_delta_pearson_macro"] = (
            float(np.mean(position_pearsons)) if position_pearsons else 0.0
        )
        metrics[f"{prefix}_position_conditioned_delta_pearson_weighted"] = weighted_mean(
            position_pearsons, position_weights
        )
        metrics[f"{prefix}_position_conditioned_delta_spearman_macro"] = (
            float(np.mean(position_spearmans)) if position_spearmans else 0.0
        )
        metrics[f"{prefix}_position_conditioned_delta_spearman_weighted"] = weighted_mean(
            position_spearmans, position_weights
        )

        for turn_idx in sorted({int(row.get("source_turn_idx", 0)) for row in rows}):
            turn_rows = [row for row in rows if int(row.get("source_turn_idx", 0)) == turn_idx]
            turn_entropy = [float(row[field]) for row in turn_rows]
            turn_delta = [float(row["ig_delta"]) for row in turn_rows]
            metrics[f"{prefix}_turn{turn_idx + 1}_pair_count"] = float(len(turn_rows))
            metrics[f"{prefix}_turn{turn_idx + 1}_delta_pearson"] = pearson(
                turn_entropy, turn_delta
            )
            metrics[f"{prefix}_turn{turn_idx + 1}_delta_spearman"] = pearson(
                average_ranks(turn_entropy), average_ranks(turn_delta)
            )

        residual_rows = [
            row
            for row in rows
            if row.get(token_count_field) is not None
            and np.isfinite(float(row[token_count_field]))
            and np.isfinite(float(row.get("source_relative_turn", 0.0)))
        ]
        residual_entropy = [float(row[field]) for row in residual_rows]
        residual_delta = [float(row["ig_delta"]) for row in residual_rows]
        residual_features = [
            [float(row.get("source_relative_turn", 0.0)), float(row[token_count_field])]
            for row in residual_rows
        ]
        entropy_residual = residualize(residual_entropy, residual_features)
        delta_residual = residualize(residual_delta, residual_features)
        residual_rank_features = [
            list(values)
            for values in zip(
                average_ranks([features[0] for features in residual_features]),
                average_ranks([features[1] for features in residual_features]),
                strict=True,
            )
        ]
        entropy_rank_residual = residualize(
            average_ranks(residual_entropy), residual_rank_features
        )
        delta_rank_residual = residualize(average_ranks(residual_delta), residual_rank_features)
        metrics[f"{prefix}_position_length_residual_pair_count"] = float(len(entropy_residual))
        metrics[f"{prefix}_position_length_residual_delta_pearson"] = pearson(
            entropy_residual, delta_residual
        )
        metrics[f"{prefix}_position_length_residual_delta_spearman"] = pearson(
            entropy_rank_residual, delta_rank_residual
        )

        # Compare entropy against the much cheaper last-turn heuristic. Require
        # at least two non-last candidates so last+entropy still saves work.
        last_control_groups: list[list[dict[str, Any]]] = []
        for group in multi_groups:
            nonlast = [row for row in group if not is_last(row)]
            if len(nonlast) >= 2 and any(is_last(row) for row in group):
                last_control_groups.append(group)
        last_only_matches: list[float] = []
        last_plus_entropy_matches: list[float] = []
        entropy_matches_when_delta_nonlast: list[float] = []
        nonlast_entropy_matches_when_delta_nonlast: list[float] = []
        for group in last_control_groups:
            delta_best = max(group, key=lambda row: float(row["ig_delta"]))
            entropy_best = max(group, key=lambda row: float(row[field]))
            nonlast_entropy_best = max(
                (row for row in group if not is_last(row)),
                key=lambda row: float(row[field]),
            )
            delta_best_is_last = is_last(delta_best)
            last_only_matches.append(float(delta_best_is_last))
            last_plus_entropy_matches.append(
                float(delta_best_is_last or delta_best is nonlast_entropy_best)
            )
            if not delta_best_is_last:
                entropy_matches_when_delta_nonlast.append(float(entropy_best is delta_best))
                nonlast_entropy_matches_when_delta_nonlast.append(
                    float(nonlast_entropy_best is delta_best)
                )
        last_only_match_frac = float(np.mean(last_only_matches)) if last_only_matches else 0.0
        last_plus_match_frac = (
            float(np.mean(last_plus_entropy_matches)) if last_plus_entropy_matches else 0.0
        )
        metrics[f"{prefix}_last_control_prompt_count"] = float(len(last_control_groups))
        metrics[f"{prefix}_last_only_max_delta_match_frac"] = last_only_match_frac
        metrics[f"{prefix}_last_plus_entropy_nonlast_top1_recall"] = last_plus_match_frac
        metrics[f"{prefix}_last_plus_entropy_nonlast_top1_recall_lift"] = (
            last_plus_match_frac - last_only_match_frac
        )
        metrics[f"{prefix}_max_delta_nonlast_prompt_count"] = float(
            len(entropy_matches_when_delta_nonlast)
        )
        metrics[f"{prefix}_top1_match_given_max_delta_nonlast"] = (
            float(np.mean(entropy_matches_when_delta_nonlast))
            if entropy_matches_when_delta_nonlast
            else 0.0
        )
        metrics[f"{prefix}_nonlast_top1_match_given_max_delta_nonlast"] = (
            float(np.mean(nonlast_entropy_matches_when_delta_nonlast))
            if nonlast_entropy_matches_when_delta_nonlast
            else 0.0
        )

    metrics: dict[str, float] = {}
    add_signal(
        "igsd/m2_action_entropy",
        "student_action_entropy",
        "student_action_token_count",
        metrics,
    )
    add_signal(
        "igsd/m2_query_entropy",
        "student_query_entropy",
        "student_query_token_count",
        metrics,
    )
    add_signal(
        "igsd/m2_non_query_entropy",
        "student_non_query_entropy",
        "student_non_query_token_count",
        metrics,
    )
    return metrics


def compute_paired_query_ig(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    actor_rollout_wg: Any,
    config: Any,
    global_step: int,
    log_prob_micro_batch_size_per_gpu: int | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute teacher and student IG with a shared prefix and document controls."""

    started = time.perf_counter()
    num_cf = max(int(config.get("igsd_num_counterfactual", 1) or 1), 1)
    gate_mode = str(config.get("igsd_gate_mode", "sigmoid")).lower()
    lcb_kappa = float(config.get("igsd_lcb_kappa", 1.0) or 0.0)
    lcb_min_counterfactuals = max(int(config.get("igsd_lcb_min_counterfactuals", 3) or 3), 2)
    effective_margin = effective_gate_margin(config, global_step)
    max_answer_tokens = max(int(config.get("igsd_max_answer_tokens", 128) or 128), 1)
    max_pseudo_seq_len = max(int(config.get("igsd_max_pseudo_seq_len", 0) or 0), 0)
    answer_template = str(config.get("igsd_answer_template", "\n<answer>{answer}</answer>"))
    doc_pool = _dedup_docs(
        [
            doc
            for ex in examples
            for doc in ex.get("student_documents", [])
            + ex.get("teacher_documents", [])
            + [
                query_doc
                for query_docs in ex.get("student_query_documents", [])
                for query_doc in query_docs
            ]
            + [
                query_doc
                for query_docs in ex.get("teacher_query_documents", [])
                for query_doc in query_docs
            ]
        ]
    )
    rng = np.random.default_rng(17_071 + int(global_step))

    prompts: list[list[int]] = []
    answers: list[list[int]] = []
    mapping: list[tuple[int, str, int]] = []
    valid_examples: list[dict[str, Any]] = []
    skipped_long_pseudo = 0
    skipped_empty_doc_pool = 0
    for ex in examples:
        aliases = ex["answer_aliases"]
        student_documents = _dedup_docs(ex.get("student_documents", []))
        teacher_documents = _dedup_docs(ex.get("teacher_documents", []))
        student_query_documents = [
            _dedup_docs(documents) for documents in ex.get("student_query_documents", [])
        ]
        teacher_query_documents = [
            _dedup_docs(documents) for documents in ex.get("teacher_query_documents", [])
        ]
        if not student_query_documents and student_documents:
            student_query_documents = [student_documents]
        if not teacher_query_documents and teacher_documents:
            teacher_query_documents = [teacher_documents]
        if not aliases or not student_documents or not teacher_documents:
            continue
        alias_ids = [
            tokenizer.encode(answer_template.format(answer=alias), add_special_tokens=False)[:max_answer_tokens]
            for alias in aliases
        ]
        alias_ids = [ids for ids in alias_ids if ids]
        if not alias_ids:
            continue

        student_queries = [
            str(query).strip()
            for query in ex.get("student_queries", [ex.get("student_query", "")])
            if str(query).strip()
        ]
        teacher_queries = [
            str(query).strip()
            for query in ex.get("teacher_queries", [ex.get("teacher_query", "")])
            if str(query).strip()
        ]
        if len(student_query_documents) < len(student_queries):
            student_query_documents.extend([[] for _ in range(len(student_queries) - len(student_query_documents))])
        elif len(student_query_documents) > len(student_queries):
            student_query_documents = student_query_documents[: len(student_queries)]
        if len(teacher_query_documents) < len(teacher_queries):
            teacher_query_documents.extend([[] for _ in range(len(teacher_queries) - len(teacher_query_documents))])
        elif len(teacher_query_documents) > len(teacher_queries):
            teacher_query_documents = teacher_query_documents[: len(teacher_queries)]
        student_action = tokenizer.encode(canonical_search_action_queries(student_queries), add_special_tokens=False)
        teacher_action = tokenizer.encode(canonical_search_action_queries(teacher_queries), add_special_tokens=False)
        # Build every real/counterfactual branch through the same synthetic
        # tool-response formatter. This avoids comparing student-real logprob
        # from original rollout IDs against teacher/counterfactual logprob from
        # reconstructed IDs.
        student_real_response_ids = _synthetic_tool_response_ids_by_query(
            student_query_documents,
            ex["tool_response_template_text"],
            tokenizer,
            ex["max_tool_response_tokens"],
            ex["tool_response_truncate_side"],
            ex.get("tool_response_result_separator", "\n-*-*-\n"),
        )
        student_real = ex["prefix_ids"] + student_action + student_real_response_ids
        teacher_real = ex["prefix_ids"] + teacher_action + _synthetic_tool_response_ids_by_query(
            teacher_query_documents,
            ex["tool_response_template_text"],
            tokenizer,
            ex["max_tool_response_tokens"],
            ex["tool_response_truncate_side"],
            ex.get("tool_response_result_separator", "\n-*-*-\n"),
        )

        local_idx = len(valid_examples)
        valid_examples.append(ex)
        contexts: list[tuple[str, int, list[int]]] = [
            ("student_real", 0, student_real),
            ("teacher_real", 0, teacher_real),
        ]
        excluded = {
            doc.strip()
            for documents in [
                student_documents,
                teacher_documents,
                *student_query_documents,
                *teacher_query_documents,
            ]
            for doc in documents
            if doc.strip()
        }
        query_count = max(len(student_query_documents), len(teacher_query_documents), 1)
        n_docs = max(
            [
                *[len(documents) for documents in student_query_documents],
                *[len(documents) for documents in teacher_query_documents],
                1,
            ]
        )
        for cf_idx in range(num_cf):
            random_docs = [
                _sample_shared_docs(doc_pool, excluded, n_docs, rng)
                for _ in range(query_count)
            ]
            if not any(random_docs):
                skipped_empty_doc_pool += 1
                continue
            student_random_response_ids = _synthetic_tool_response_ids_by_query(
                random_docs[: len(student_queries)],
                ex["tool_response_template_text"],
                tokenizer,
                ex["max_tool_response_tokens"],
                ex["tool_response_truncate_side"],
                ex.get("tool_response_result_separator", "\n-*-*-\n"),
            )
            teacher_random_response_ids = _synthetic_tool_response_ids_by_query(
                random_docs[: len(teacher_queries)],
                ex["tool_response_template_text"],
                tokenizer,
                ex["max_tool_response_tokens"],
                ex["tool_response_truncate_side"],
                ex.get("tool_response_result_separator", "\n-*-*-\n"),
            )
            contexts.append(
                ("student_cf", cf_idx, ex["prefix_ids"] + student_action + student_random_response_ids)
            )
            contexts.append(
                ("teacher_cf", cf_idx, ex["prefix_ids"] + teacher_action + teacher_random_response_ids)
            )

        for kind, cf_idx, context in contexts:
            for answer_ids in alias_ids:
                if max_pseudo_seq_len > 0 and len(context) + len(answer_ids) > max_pseudo_seq_len:
                    skipped_long_pseudo += 1
                    continue
                prompts.append(context)
                answers.append(answer_ids)
                mapping.append((local_idx, kind, cf_idx))

    if not prompts:
        empty_examples, routing_metrics = apply_turn_routing(
            [],
            mode=str(config.get("igsd_turn_routing_mode", "independent")).lower(),
            gate_margin=effective_margin,
        )
        del empty_examples
        metrics = {
            "igsd/m2_input_pair_count": float(len(examples)),
            "igsd/m2_valid_pair_count": 0.0,
            "igsd/m2_pair_coverage": 0.0,
            "igsd/m2_pseudo_sequence_count": 0.0,
            "igsd/m2_skipped_long_pseudo_count": float(skipped_long_pseudo),
            "igsd/m2_skipped_empty_doc_pool_count": float(skipped_empty_doc_pool),
            "igsd/m2_doc_pool_size": float(len(doc_pool)),
            "igsd/m2_effective_gate_margin": effective_margin,
            "igsd/m2_gate_mode_is_lcb": float(gate_mode == "lcb_sigmoid"),
            "igsd/m2_lcb_kappa": lcb_kappa,
            "igsd/m2_lcb_min_counterfactuals": float(lcb_min_counterfactuals),
            "igsd/m2_counterfactual_count_mean": 0.0,
            "igsd/m2_counterfactual_count_min": 0.0,
            "igsd/m2_counterfactual_count_max": 0.0,
            "igsd/m2_cf_delta_std_mean": 0.0,
            "igsd/m2_cf_delta_se_mean": 0.0,
            "igsd/m2_cf_delta_positive_frac_mean": 0.0,
            "igsd/m2_cf_delta_sign_agreement_mean": 0.0,
            "igsd/m2_lcb_delta_mean": 0.0,
            "igsd/m2_lcb_positive_frac": 0.0,
            "igsd/m2_lcb_valid_frac": 0.0,
            "igsd/m2_lcb_insufficient_pair_count": 0.0,
            "igsd/m2_gate_effective_sample_size": 0.0,
            "igsd/m2_gate_effective_sample_frac": 0.0,
            "igsd/m2_time_sec": time.perf_counter() - started,
        }
        metrics.update(routing_metrics)
        metrics.update(entropy_delta_metrics([]))
        return [], metrics

    # Bound activation memory while preserving vectorized worker calls. Each
    # chunk is padded internally to world_size * micro_batch_size_per_gpu.
    log_probs: list[float] = []
    max_pseudo_batch = max(int(config.get("igsd_max_pseudo_batch", 256) or 256), 1)
    for start in range(0, len(prompts), max_pseudo_batch):
        chunk_log_probs, _ = _call_compute_log_prob(
            prompts[start : start + max_pseudo_batch],
            answers[start : start + max_pseudo_batch],
            tokenizer,
            actor_rollout_wg,
            log_prob_micro_batch_size_per_gpu,
        )
        log_probs.extend(chunk_log_probs)
    scores: dict[tuple[int, str, int], list[float]] = {}
    for key, log_prob in zip(mapping, log_probs, strict=True):
        scores.setdefault(key, []).append(float(log_prob))

    completed: list[dict[str, Any]] = []
    for local_idx, ex in enumerate(valid_examples):
        student_real_values = scores.get((local_idx, "student_real", 0), [])
        teacher_real_values = scores.get((local_idx, "teacher_real", 0), [])
        student_cf_by_idx = {
            cf_idx: max(values)
            for (idx, kind, cf_idx), values in scores.items()
            if idx == local_idx and kind == "student_cf" and values
        }
        teacher_cf_by_idx = {
            cf_idx: max(values)
            for (idx, kind, cf_idx), values in scores.items()
            if idx == local_idx and kind == "teacher_cf" and values
        }
        paired_cf_indices = sorted(student_cf_by_idx.keys() & teacher_cf_by_idx.keys())
        if not student_real_values or not teacher_real_values or not paired_cf_indices:
            continue
        student_real = max(student_real_values)
        teacher_real = max(teacher_real_values)
        student_cf = [student_cf_by_idx[idx] for idx in paired_cf_indices]
        teacher_cf = [teacher_cf_by_idx[idx] for idx in paired_cf_indices]
        ig_student = student_real - float(np.mean(student_cf))
        ig_teacher = teacher_real - float(np.mean(teacher_cf))
        cf_deltas = np.asarray(
            [
                (teacher_real - teacher_cf_by_idx[idx]) - (student_real - student_cf_by_idx[idx])
                for idx in paired_cf_indices
            ],
            dtype=np.float64,
        )
        delta = float(cf_deltas.mean())
        cf_delta_std = float(cf_deltas.std(ddof=1)) if len(cf_deltas) > 1 else 0.0
        cf_delta_se = float(cf_delta_std / np.sqrt(len(cf_deltas)))
        cf_positive_frac = float(np.mean(cf_deltas > 0.0))
        cf_sign_agreement = max(cf_positive_frac, 1.0 - cf_positive_frac)
        lcb_delta = float(delta - lcb_kappa * cf_delta_se)
        lcb_valid = len(cf_deltas) >= lcb_min_counterfactuals
        if gate_mode == "lcb_sigmoid" and not lcb_valid:
            gate = 0.0
        else:
            gate = float(
                compute_gate(
                    torch.tensor([delta], dtype=torch.float32),
                    config,
                    global_step,
                    delta_standard_error=torch.tensor([cf_delta_se], dtype=torch.float32),
                ).item()
            )
        gate_signal = lcb_delta if gate_mode == "lcb_sigmoid" else delta
        completed.append(
            {
                **ex,
                "ig_student": ig_student,
                "ig_teacher": ig_teacher,
                "ig_delta": delta,
                "ig_gate_signal": gate_signal,
                "ig_gate": gate,
                "ig_cf_count": len(cf_deltas),
                "ig_cf_delta_std": cf_delta_std,
                "ig_cf_delta_se": cf_delta_se,
                "ig_cf_delta_positive_frac": cf_positive_frac,
                "ig_cf_delta_sign_agreement": cf_sign_agreement,
                "ig_lcb_delta": lcb_delta,
                "ig_lcb_valid": lcb_valid,
            }
        )

    routing_mode = str(config.get("igsd_turn_routing_mode", "independent")).lower()
    completed, routing_metrics = apply_turn_routing(
        completed,
        mode=routing_mode,
        gate_margin=effective_margin,
    )

    def mean(key: str) -> float:
        return float(np.mean([ex[key] for ex in completed])) if completed else 0.0

    metrics = {
        "igsd/m2_input_pair_count": float(len(examples)),
        "igsd/m2_valid_pair_count": float(len(completed)),
        "igsd/m2_pair_coverage": float(len(completed) / max(len(examples), 1)),
        "igsd/m2_pseudo_sequence_count": float(len(prompts)),
        "igsd/m2_skipped_long_pseudo_count": float(skipped_long_pseudo),
        "igsd/m2_skipped_empty_doc_pool_count": float(skipped_empty_doc_pool),
        "igsd/m2_doc_pool_size": float(len(doc_pool)),
        "igsd/m2_effective_gate_margin": effective_margin,
        "igsd/m2_gate_mode_is_lcb": float(gate_mode == "lcb_sigmoid"),
        "igsd/m2_lcb_kappa": lcb_kappa,
        "igsd/m2_lcb_min_counterfactuals": float(lcb_min_counterfactuals),
        "igsd/m2_counterfactual_count_mean": mean("ig_cf_count"),
        "igsd/m2_counterfactual_count_min": float(
            min((ex["ig_cf_count"] for ex in completed), default=0)
        ),
        "igsd/m2_counterfactual_count_max": float(
            max((ex["ig_cf_count"] for ex in completed), default=0)
        ),
        "igsd/m2_cf_delta_std_mean": mean("ig_cf_delta_std"),
        "igsd/m2_cf_delta_se_mean": mean("ig_cf_delta_se"),
        "igsd/m2_cf_delta_positive_frac_mean": mean("ig_cf_delta_positive_frac"),
        "igsd/m2_cf_delta_sign_agreement_mean": mean("ig_cf_delta_sign_agreement"),
        "igsd/m2_lcb_delta_mean": mean("ig_lcb_delta"),
        "igsd/m2_lcb_positive_frac": float(
            np.mean(
                [ex["ig_lcb_delta"] > effective_margin for ex in completed if ex["ig_lcb_valid"]]
            )
            if any(ex["ig_lcb_valid"] for ex in completed)
            else 0.0
        ),
        "igsd/m2_lcb_valid_frac": float(
            np.mean([ex["ig_lcb_valid"] for ex in completed]) if completed else 0.0
        ),
        "igsd/m2_lcb_insufficient_pair_count": float(
            sum(not ex["ig_lcb_valid"] for ex in completed)
        ),
        "igsd/m2_forward_chunk_count": float((len(prompts) + max_pseudo_batch - 1) // max_pseudo_batch),
        "igsd/m2_ig_student_mean": mean("ig_student"),
        "igsd/m2_ig_teacher_mean": mean("ig_teacher"),
        "igsd/m2_ig_delta_mean": mean("ig_delta"),
        "igsd/m2_gate_mean": mean("ig_gate"),
        "igsd/m2_positive_delta_frac": float(
            np.mean([ex["ig_delta"] > 0 for ex in completed]) if completed else 0.0
        ),
        "igsd/m2_teacher_query_changed_frac": float(
            np.mean([ex["teacher_query"].strip() != ex["student_query"].strip() for ex in completed])
            if completed
            else 0.0
        ),
        "igsd/m2_answer_alias_count_mean": float(
            np.mean([len(ex["answer_aliases"]) for ex in completed]) if completed else 0.0
        ),
        "igsd/m2_time_sec": time.perf_counter() - started,
    }
    metrics.update(routing_metrics)
    metrics.update(turn_level_ig_metrics(completed))
    metrics.update(entropy_delta_metrics(completed))
    if completed:
        deltas = np.asarray([ex["ig_delta"] for ex in completed], dtype=np.float32)
        gates = np.asarray([ex["ig_gate"] for ex in completed], dtype=np.float32)
        metrics.update(
            {
                "igsd/m2_ig_delta_std": float(deltas.std()),
                "igsd/m2_ig_delta_min": float(deltas.min()),
                "igsd/m2_ig_delta_max": float(deltas.max()),
                "igsd/m2_gate_std": float(gates.std()),
                "igsd/m2_gate_above_half_frac": float(np.mean(gates > 0.5)),
                "igsd/m2_gate_effective_sample_size": float(
                    gates.sum() ** 2 / max(float(np.square(gates).sum()), np.finfo(np.float32).eps)
                ),
                "igsd/m2_gate_effective_sample_frac": float(
                    (gates.sum() ** 2 / max(float(np.square(gates).sum()), np.finfo(np.float32).eps))
                    / len(gates)
                ),
            }
        )
    else:
        metrics.update(
            {
                "igsd/m2_ig_delta_std": 0.0,
                "igsd/m2_ig_delta_min": 0.0,
                "igsd/m2_ig_delta_max": 0.0,
                "igsd/m2_gate_std": 0.0,
                "igsd/m2_gate_above_half_frac": 0.0,
                "igsd/m2_gate_effective_sample_size": 0.0,
                "igsd/m2_gate_effective_sample_frac": 0.0,
            }
        )
    return completed, metrics


def _matched_branch_pair_ig(
    left_real: dict[int, float] | None,
    right_real: dict[int, float] | None,
    left_cf: dict[int, dict[int, float]] | None,
    right_cf: dict[int, dict[int, float]] | None,
    alias_count: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a four-cell, shared-counterfactual branch IG comparison."""

    if left_real is None or right_real is None or left_cf is None or right_cf is None:
        return None, "missing_branch_scores"
    shared_cf = sorted(left_cf.keys() & right_cf.keys())
    if not shared_cf:
        return None, "disjoint_counterfactuals"
    left_deltas: list[float] = []
    right_deltas: list[float] = []
    common_alias_counts: list[int] = []
    common_alias_full_coverage: list[bool] = []
    for cf_idx in shared_cf:
        common_aliases = sorted(
            left_real.keys()
            & right_real.keys()
            & left_cf[cf_idx].keys()
            & right_cf[cf_idx].keys()
        )
        if not common_aliases:
            continue
        left_delta = float(
            max(left_real[alias_idx] for alias_idx in common_aliases)
            - max(left_cf[cf_idx][alias_idx] for alias_idx in common_aliases)
        )
        right_delta = float(
            max(right_real[alias_idx] for alias_idx in common_aliases)
            - max(right_cf[cf_idx][alias_idx] for alias_idx in common_aliases)
        )
        if not np.isfinite(left_delta) or not np.isfinite(right_delta):
            continue
        left_deltas.append(left_delta)
        right_deltas.append(right_delta)
        common_alias_counts.append(len(common_aliases))
        common_alias_full_coverage.append(len(common_aliases) == alias_count)
    if not left_deltas:
        return None, "no_common_aliases"
    left_ig = float(np.mean(left_deltas))
    right_ig = float(np.mean(right_deltas))
    gain = float(np.mean(np.asarray(left_deltas) - np.asarray(right_deltas)))
    if not np.isfinite(left_ig) or not np.isfinite(right_ig) or not np.isfinite(gain):
        return None, "nonfinite_pair_delta"
    return {
        "left_ig": left_ig,
        "right_ig": right_ig,
        "gain": gain,
        "common_cf_count": len(left_deltas),
        "common_alias_count_min": min(common_alias_counts),
        "common_alias_count_mean": float(np.mean(common_alias_counts)),
        "common_alias_counts": common_alias_counts,
        "common_alias_full_coverage": common_alias_full_coverage,
    }, None


def _apply_sampled_pair_non_environment_gate(
    examples: list[dict[str, Any]],
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Attach no-environment gate signals to routed sampled-action pairs.

    The planner has already materialized the same teacher/reference pair
    targets and budget decisions used by the environment-verified path.  This
    helper deliberately performs no continuation generation, retrieval, or
    actor answer scoring.  It is therefore suitable for clean internal-signal
    and constant-weight controls rather than an environment-gate fallback.
    """

    started = time.perf_counter()
    source = str(config.get("igsd_sampled_pair_gate_source", "environment_ig")).lower()
    granularity = str(
        config.get("igsd_sampled_pair_gate_granularity", "token")
    ).lower()
    if source not in {"likelihood_gap", "constant"}:
        raise ValueError(
            "Non-environment sampled-pair helper requires gate_source in "
            "{'likelihood_gap', 'constant'}, got "
            f"{source!r}"
        )
    if granularity != "token":
        raise ValueError(
            "Non-environment sampled-pair gates require gate_granularity='token', "
            f"got {granularity!r}"
        )
    if source == "likelihood_gap" and str(
        config.get("igsd_sampled_pair_candidate_mode", "sampled_action")
    ).lower() != "sampled_action":
        raise ValueError(
            "likelihood_gap sampled-pair gating requires candidate_mode='sampled_action'"
        )
    unexpected_branch_count = sum(
        len(ex.get("igsd_token_branch_records", [])) for ex in examples
    )
    if unexpected_branch_count:
        raise ValueError(
            "Non-environment sampled-pair gating must not receive continuation branches, "
            f"got {unexpected_branch_count}"
        )

    margin = float(config.get("igsd_token_gate_margin", 0.0))
    if not math.isfinite(margin):
        raise ValueError(f"igsd_token_gate_margin must be finite, got {margin}")
    total_query_tokens = 0
    selected_count = 0
    valid_count = 0
    active_count = 0
    all_signals: list[float] = []
    all_gates: list[float] = []
    all_valid_gates: list[float] = []

    for ex in examples:
        query_token_count = sum(bool(value) for value in ex.get("student_action_query_mask", []))
        plans = ex.get("igsd_token_budget_plan", [])
        if len(plans) != query_token_count:
            raise ValueError(
                "Non-environment sampled-pair plans must align with sampled query tokens, got "
                f"{len(plans)=} and {query_token_count=}"
            )
        total_query_tokens += query_token_count
        signals = np.zeros(query_token_count, dtype=np.float32)
        signal_valid = np.zeros(query_token_count, dtype=bool)
        for token_idx, plan in enumerate(plans):
            plan["pair_gate_source"] = source
            plan["pair_gate_valid"] = False
            plan["pair_gate_signal"] = 0.0
            plan["pair_gate"] = 0.0
            plan["pair_active"] = False
            plan["pair_positive_gain"] = False
            plan["pair_ig_valid"] = False
            plan["pair_scoreable"] = False
            plan.pop("pair_gain", None)
            plan.pop("pair_gate_error", None)
            if not bool(plan.get("selected", False)):
                continue
            selected_count += 1
            if source == "constant":
                signal = 1.0
            else:
                teacher_sample = plan.get("teacher_sample_log_prob")
                student_sample = plan.get("student_sample_log_prob")
                if teacher_sample is None or student_sample is None:
                    plan["pair_gate_error"] = "missing_sample_log_probs"
                    continue
                signal = float(teacher_sample) - float(student_sample)
                if not math.isfinite(signal):
                    plan["pair_gate_error"] = "nonfinite_likelihood_gap"
                    continue
            signals[token_idx] = float(signal)
            signal_valid[token_idx] = True
            plan["pair_gate_valid"] = True
            plan["pair_gate_signal"] = float(signal)

        if source == "constant":
            gates = signal_valid.astype(np.float32)
        else:
            gates, _ = _token_gate_and_row_normalize(signals, signal_valid, config)
            gates[~(signal_valid & (signals > margin))] = 0.0
        active = signal_valid & (gates > 0.0)
        for token_idx, plan in enumerate(plans):
            plan["pair_gate"] = float(gates[token_idx])
            plan["pair_active"] = bool(active[token_idx])
        valid_count += int(signal_valid.sum())
        active_count += int(active.sum())
        all_signals.extend(signals[signal_valid].astype(float).tolist())
        all_gates.extend(gates.astype(float).tolist())
        all_valid_gates.extend(gates[signal_valid].astype(float).tolist())
        ex["igsd_token_ig_values"] = []
        ex["igsd_token_branch_ig_valid"] = []
        ex["igsd_token_gains"] = signals.tolist()
        ex["igsd_token_gate_signals"] = signals.tolist()
        ex["igsd_token_gate"] = gates.astype(np.float32).tolist()
        ex["igsd_token_delta_valid"] = signal_valid.tolist()
        # Preserve every query token in the auxiliary-loss denominator.
        ex["igsd_token_gate_valid"] = np.ones(query_token_count, dtype=bool).tolist()
        ex["igsd_token_pair_left_ig"] = [0.0] * query_token_count
        ex["igsd_token_pair_right_ig"] = [0.0] * query_token_count
        ex["igsd_token_common_cf_count"] = [0] * query_token_count
        ex["igsd_token_common_alias_count_min"] = [0] * query_token_count
        ex["igsd_token_common_alias_count_mean"] = [0.0] * query_token_count

    output_gate_values = np.asarray(all_gates, dtype=np.float64)
    valid_gate_values = np.asarray(all_valid_gates, dtype=np.float64)
    signal_values = np.asarray(all_signals, dtype=np.float64)
    metrics: dict[str, float] = {
        "igsd/token_intervention_scorer_example_count": float(len(examples)),
        "igsd/token_intervention_pseudo_sequence_count": 0.0,
        "igsd/token_intervention_forward_chunk_count": 0.0,
        "igsd/token_intervention_skipped_example_count": 0.0,
        "igsd/token_intervention_skipped_long_pseudo_count": 0.0,
        "igsd/token_intervention_skipped_empty_doc_pool_count": 0.0,
        "igsd/token_intervention_nonfinite_log_prob_count": 0.0,
        "igsd/token_intervention_doc_pool_size": 0.0,
        "igsd/token_intervention_valid_branch_count": 0.0,
        "igsd/token_intervention_valid_branch_frac": 0.0,
        "igsd/token_intervention_valid_token_count": float(valid_count),
        "igsd/token_intervention_valid_token_frac": float(
            valid_count / max(total_query_tokens, 1)
        ),
        "igsd/token_intervention_effective_token_count": float(total_query_tokens),
        "igsd/token_intervention_fallback_token_count": 0.0,
        "igsd/token_intervention_fallback_token_frac": 0.0,
        "igsd/token_intervention_skipped_branch_count": 0.0,
        "igsd/token_intervention_all_invalid_query_count": float(
            sum(
                bool(sum(bool(value) for value in ex.get("student_action_query_mask", [])))
                and not any(
                    bool(plan.get("pair_gate_valid", False))
                    for plan in ex.get("igsd_token_budget_plan", [])
                )
                for ex in examples
            )
        ),
        "igsd/token_intervention_all_zero_gate_query_count": float(
            sum(
                bool(sum(bool(value) for value in ex.get("student_action_query_mask", [])))
                and not any(float(value) > 0.0 for value in ex.get("igsd_token_gate", []))
                for ex in examples
            )
        ),
        "igsd/token_intervention_positive_gate_token_count": float(active_count),
        "igsd/token_intervention_positive_gate_token_frac": float(
            active_count / max(total_query_tokens, 1)
        ),
        "igsd/token_intervention_disjoint_cf_token_count": 0.0,
        "igsd/token_intervention_alias_disjoint_cf_count": 0.0,
        "igsd/token_intervention_no_usable_paired_cf_token_count": 0.0,
        "igsd/token_intervention_nonfinite_pair_delta_count": 0.0,
        "igsd/token_intervention_common_cf_count_mean": 0.0,
        "igsd/token_intervention_common_cf_count_min": 0.0,
        "igsd/token_intervention_common_cf_count_max": 0.0,
        "igsd/token_intervention_common_cf_full_coverage_frac": 0.0,
        "igsd/token_intervention_common_alias_count_mean": 0.0,
        "igsd/token_intervention_common_alias_count_min": 0.0,
        "igsd/token_intervention_common_alias_count_max": 0.0,
        "igsd/token_intervention_common_alias_full_coverage_frac": 0.0,
        "igsd/token_intervention_gain_mean": float(signal_values.mean()) if signal_values.size else 0.0,
        "igsd/token_intervention_gain_std": float(signal_values.std()) if signal_values.size else 0.0,
        "igsd/token_intervention_raw_gate_mean": float(valid_gate_values.mean())
        if valid_gate_values.size
        else 0.0,
        "igsd/token_intervention_raw_gate_max": float(valid_gate_values.max())
        if valid_gate_values.size
        else 0.0,
        "igsd/token_intervention_positive_gate_frac": float(
            np.mean(valid_gate_values > 0.0) if valid_gate_values.size else 0.0
        ),
        "igsd/token_intervention_gate_zero_frac": float(
            np.mean(valid_gate_values <= 0.0) if valid_gate_values.size else 0.0
        ),
        "igsd/token_intervention_normalized_gate_mean": float(output_gate_values.mean())
        if output_gate_values.size
        else 0.0,
        "igsd/token_intervention_normalized_gate_std": float(output_gate_values.std())
        if output_gate_values.size
        else 0.0,
        "igsd/token_intervention_normalized_gate_max": float(output_gate_values.max())
        if output_gate_values.size
        else 0.0,
        "igsd/token_intervention_output_gate_mean": float(output_gate_values.mean())
        if output_gate_values.size
        else 0.0,
        "igsd/token_intervention_output_gate_std": float(output_gate_values.std())
        if output_gate_values.size
        else 0.0,
        "igsd/token_intervention_output_gate_max": float(output_gate_values.max())
        if output_gate_values.size
        else 0.0,
        "igsd/token_intervention_gate_normalization_is_row_mean": float(
            str(config.get("igsd_token_gate_normalization", "row_mean")).lower() == "row_mean"
        ),
        "igsd/token_intervention_gate_normalization_is_none": float(
            str(config.get("igsd_token_gate_normalization", "row_mean")).lower() == "none"
        ),
        "igsd/token_intervention_time_sec": time.perf_counter() - started,
        "igsd/sampled_pair_gate_source_is_environment_ig": 0.0,
        "igsd/sampled_pair_gate_source_is_likelihood_gap": float(source == "likelihood_gap"),
        "igsd/sampled_pair_gate_source_is_constant": float(source == "constant"),
        "igsd/sampled_pair_gate_granularity_is_token": float(granularity == "token"),
        "igsd/sampled_pair_gate_granularity_is_query_mean": float(
            granularity == "query_mean"
        ),
        "igsd/sampled_pair_gate_selected_count": float(selected_count),
        "igsd/sampled_pair_gate_valid_count": float(valid_count),
        "igsd/sampled_pair_gate_active_count": float(active_count),
        "igsd/sampled_pair_gate_positive_frac": float(active_count / max(valid_count, 1)),
        "igsd/sampled_pair_gate_signal_mean": float(signal_values.mean()) if signal_values.size else 0.0,
        "igsd/sampled_pair_gate_signal_std": float(signal_values.std()) if signal_values.size else 0.0,
    }
    return examples, metrics


def compute_token_intervention_ig(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    actor_rollout_wg: Any,
    config: Any,
    global_step: int,
    log_prob_micro_batch_size_per_gpu: int | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Score G1 branches with the existing paired counterfactual IG proxy.

    Every branch for one query shares the same counterfactual document samples
    and the same synthetic-response formatting. For each adjacent pair and CF,
    gains use only aliases with finite scores in all four cells (left/right real
    and left/right CF). Thus length or numerical filtering cannot introduce an
    unmatched-control or unmatched-alias difference. ``n_docs`` is fixed to the
    largest real branch document count, including the student terminal branch,
    so response length cannot become a hidden branch-dependent signal. Student,
    qT-teacher, and intervention-branch real documents are all excluded from the
    random pool for that query. These controls intentionally mirror the current
    paired evaluator.
    """

    started = time.perf_counter()
    token_weight_mode = str(config.get("igsd_token_weight_mode", "prefix_intervention_ig")).lower()
    budgeted_local_candidates = token_weight_mode == "budgeted_local_candidate_ig"
    sampled_action_pairs = token_weight_mode == "sampled_action_pair_ig"
    sampled_pair_reference_mode = str(
        config.get("igsd_sampled_pair_reference_mode", "greedy_pair")
    ).lower()
    if sampled_action_pairs and sampled_pair_reference_mode not in {"greedy_pair", "fixed_s"}:
        raise ValueError(
            "Sampled-action pair scorer reference mode must be 'greedy_pair' or 'fixed_s', "
            f"got {sampled_pair_reference_mode!r}"
        )
    sampled_pair_gate_source = str(
        config.get("igsd_sampled_pair_gate_source", "environment_ig")
    ).lower()
    sampled_pair_gate_granularity = str(
        config.get("igsd_sampled_pair_gate_granularity", "token")
    ).lower()
    if sampled_action_pairs and sampled_pair_gate_granularity not in {"token", "query_mean"}:
        raise ValueError(
            "Sampled-action pair gate granularity must be 'token' or 'query_mean', "
            f"got {sampled_pair_gate_granularity!r}"
        )
    if (
        sampled_action_pairs
        and sampled_pair_gate_granularity == "query_mean"
        and sampled_pair_gate_source != "environment_ig"
    ):
        raise ValueError(
            "Sampled-action pair query_mean gate granularity requires gate_source='environment_ig'"
        )
    if sampled_action_pairs and sampled_pair_gate_source in {"likelihood_gap", "constant"}:
        return _apply_sampled_pair_non_environment_gate(examples, config)
    num_cf = max(int(config.get("igsd_num_counterfactual", 1) or 1), 1)
    max_answer_tokens = max(int(config.get("igsd_max_answer_tokens", 128) or 128), 1)
    max_pseudo_seq_len = max(int(config.get("igsd_max_pseudo_seq_len", 0) or 0), 0)
    answer_template = str(config.get("igsd_answer_template", "\n<answer>{answer}</answer>"))
    rng = np.random.default_rng(31_719 + int(global_step))
    doc_pool = _dedup_docs(
        [
            doc
            for ex in examples
            for doc in ex.get("student_documents", [])
            + ex.get("teacher_documents", [])
            + [
                branch_doc
                for record in ex.get("igsd_token_branch_records", [])
                for branch_doc in record.get("branch_documents", [])
                + [
                    query_doc
                    for query_docs in record.get("branch_query_documents", [])
                    for query_doc in query_docs
                ]
            ]
        ]
    )

    prompts: list[list[int]] = []
    answers: list[list[int]] = []
    mapping: list[tuple[int, int, str, int, int]] = []
    branch_counts: list[int] = []
    query_token_counts: list[int] = []
    alias_counts = [0] * len(examples)
    attempted_real_branches: set[tuple[int, int]] = set()
    attempted_cf_by_branch: dict[tuple[int, int], set[int]] = {}
    skipped_examples = 0
    skipped_branches = 0
    skipped_long_pseudo = 0
    skipped_empty_doc_pool = 0
    nonfinite_log_prob_count = 0

    for example_idx, ex in enumerate(examples):
        records = ex.get("igsd_token_branch_records", [])
        branch_counts.append(len(records))
        aliases = ex.get("answer_aliases", [])
        query_token_count = sum(bool(value) for value in ex.get("student_action_query_mask", []))
        # Sampled-pair coverage is defined over every sampled query token,
        # including rows for which routing produced no continuation branches.
        # Keep the legacy append point for every other mode so their diagnostic
        # denominators remain byte-for-byte unchanged.
        if sampled_action_pairs:
            query_token_counts.append(query_token_count)
        if not records or not aliases:
            skipped_examples += 1
            continue
        if not sampled_action_pairs:
            query_token_counts.append(query_token_count)
        branch_indices = [int(record.get("branch_index", -1)) for record in records]
        if branch_indices != list(range(len(records))):
            raise ValueError(
                "Token-intervention branch indices must be contiguous B_0...B_n, "
                f"got {branch_indices}"
            )
        if not budgeted_local_candidates and not sampled_action_pairs and len(records) != query_token_count + 1:
            raise ValueError(
                "Token-intervention branch count must equal query token count + terminal, "
                f"got {len(records)} branches for {query_token_count} query tokens"
            )
        if budgeted_local_candidates:
            plans = ex.get("igsd_token_budget_plan", [])
            if len(plans) != query_token_count:
                raise ValueError(
                    "Budgeted local candidate plan must align with sampled query tokens, "
                    f"got {len(plans)} plans for {query_token_count} query tokens"
                )
            teacher_candidate_count = sum(
                record.get("branch_kind") in {"budgeted_teacher_top1", "budgeted_teacher_top2"}
                for record in records
            )
            if teacher_candidate_count > query_token_count:
                raise AssertionError(
                    "Budgeted local candidate scorer received more teacher branches than query tokens"
                )
        if sampled_action_pairs:
            plans = ex.get("igsd_token_budget_plan", [])
            if len(plans) != query_token_count:
                raise ValueError(
                    "Sampled-action pair plan must align with sampled query tokens, "
                    f"got {len(plans)} plans for {query_token_count} query tokens"
                )
            pair_records = [
                record
                for record in records
                if record.get("branch_kind")
                in {"sampled_pair_teacher", "sampled_pair_student"}
            ]
            if len(pair_records) > 2 * query_token_count:
                raise AssertionError(
                    "Sampled-action pair scorer received more than two branches per query token"
                )
            if sampled_pair_reference_mode == "fixed_s":
                terminal_records = [
                    record for record in records if record.get("branch_kind") == "student_terminal"
                ]
                selected_count = sum(bool(plan.get("selected", False)) for plan in plans)
                if selected_count and len(terminal_records) != 1:
                    raise ValueError(
                        "Fixed-S sampled-action pair scoring requires exactly one shared student terminal "
                        f"branch for selected pairs, got {len(terminal_records)}"
                    )
        alias_ids = [
            tokenizer.encode(answer_template.format(answer=alias), add_special_tokens=False)[:max_answer_tokens]
            for alias in aliases
        ]
        alias_ids = [ids for ids in alias_ids if ids]
        if not alias_ids:
            skipped_examples += 1
            continue
        alias_counts[example_idx] = len(alias_ids)

        # The terminal B_k=S record receives the original student documents;
        # generated branches receive retrieval documents in ray_trainer before
        # this scorer is called.  Query-slot documents are retained separately
        # when available, while the flat list remains a legacy diagnostic.
        for record in records:
            if record.get("branch_kind") == "student_terminal":
                if not record.get("branch_query_documents"):
                    flat = list(ex.get("student_documents", []))
                    query_count = len(ex.get("student_queries", []))
                    record["branch_query_documents"] = [flat] if flat else [[] for _ in range(query_count)]
                # Preserve summary-mode re-retrieval populated by ray_trainer.
                # Only fall back to the original decoded documents when no
                # branch-level evidence was attached.
                if not record.get("branch_documents"):
                    record["branch_documents"] = [
                        doc
                        for query_docs in record.get("branch_query_documents", [])
                        for doc in query_docs
                    ] or list(ex.get("student_documents", []))

        def record_queries(record: dict[str, Any]) -> list[str]:
            queries = [str(query).strip() for query in record.get("branch_queries", []) if str(query).strip()]
            if not queries:
                legacy = str(record.get("branch_query", "")).strip()
                queries = [legacy] if legacy else []
            return queries

        def record_query_documents(record: dict[str, Any]) -> list[list[str]]:
            query_documents = record.get("branch_query_documents", [])
            flat = _dedup_docs(record.get("branch_documents", []))
            if isinstance(query_documents, list) and query_documents and all(
                isinstance(docs, list) for docs in query_documents
            ) and (any(query_documents) or not flat):
                return [_dedup_docs(docs) for docs in query_documents]
            return [flat] if flat else [[] for _ in record_queries(record)]

        usable_records = [
            record
            for record in records
            if bool(record.get("branch_valid"))
            and record_queries(record)
            and _dedup_docs(record.get("branch_documents", []))
        ]
        if not usable_records:
            skipped_examples += 1
            continue
        student_documents = _dedup_docs(ex.get("student_documents", []))
        branch_documents = {
            int(record["branch_index"]): record_query_documents(record)
            for record in usable_records
        }
        student_query_documents = [
            _dedup_docs(docs)
            for docs in next(
                (
                    record_query_documents(record)
                    for record in records
                    if record.get("branch_kind") == "student_terminal"
                ),
                [student_documents],
            )
        ]
        n_docs = max(
            [
                *[len(docs) for docs in student_query_documents],
                *[len(docs) for query_docs in branch_documents.values() for docs in query_docs],
                1,
            ]
        )
        teacher_documents = _dedup_docs(ex.get("teacher_documents", []))
        excluded = {
            doc.strip()
            for docs in [
                student_documents,
                teacher_documents,
                *[docs for query_docs in branch_documents.values() for docs in query_docs],
            ]
            for doc in docs
            if doc.strip()
        }
        query_count = max(
            [
                len(student_query_documents),
                *[len(record_queries(record)) for record in usable_records],
                1,
            ]
        )
        random_docs_by_cf: dict[int, list[list[str]]] = {}
        for cf_idx in range(num_cf):
            random_docs = [
                _sample_shared_docs(doc_pool, excluded, n_docs, rng)
                for _ in range(query_count)
            ]
            if not any(random_docs):
                skipped_empty_doc_pool += 1
                continue
            random_docs_by_cf[cf_idx] = random_docs

        if not random_docs_by_cf:
            skipped_examples += 1
            continue

        prefix_ids = list(ex.get("prefix_ids", []))
        for record in usable_records:
            branch_idx = int(record["branch_index"])
            queries = record_queries(record)
            query_docs = branch_documents[branch_idx]
            if len(query_docs) < len(queries):
                query_docs = query_docs + [[] for _ in range(len(queries) - len(query_docs))]
            elif len(query_docs) > len(queries):
                query_docs = query_docs[: len(queries)]
            attempted_real_branches.add((example_idx, branch_idx))
            attempted_cf_by_branch[(example_idx, branch_idx)] = set(random_docs_by_cf)
            action_ids = tokenizer.encode(canonical_search_action_queries(queries), add_special_tokens=False)
            real_context = prefix_ids + action_ids + _synthetic_tool_response_ids_by_query(
                query_docs,
                ex["tool_response_template_text"],
                tokenizer,
                ex["max_tool_response_tokens"],
                ex["tool_response_truncate_side"],
                ex.get("tool_response_result_separator", "\n-*-*-\n"),
            )
            # Emit the real branch exactly once, then one paired control per
            # sampled counterfactual. All branches for this query share the
            # same random document list at each cf index.
            for alias_idx, answer_ids in enumerate(alias_ids):
                if max_pseudo_seq_len > 0 and len(real_context) + len(answer_ids) > max_pseudo_seq_len:
                    skipped_long_pseudo += 1
                    continue
                prompts.append(real_context)
                answers.append(answer_ids)
                mapping.append((example_idx, branch_idx, "real", 0, alias_idx))
            for cf_idx, random_docs in random_docs_by_cf.items():
                context = (
                    prefix_ids
                    + action_ids
                    + _synthetic_tool_response_ids_by_query(
                        random_docs[: len(queries)],
                        ex["tool_response_template_text"],
                        tokenizer,
                        ex["max_tool_response_tokens"],
                        ex["tool_response_truncate_side"],
                        ex.get("tool_response_result_separator", "\n-*-*-\n"),
                    )
                )
                for alias_idx, answer_ids in enumerate(alias_ids):
                    if max_pseudo_seq_len > 0 and len(context) + len(answer_ids) > max_pseudo_seq_len:
                        skipped_long_pseudo += 1
                        continue
                    prompts.append(context)
                    answers.append(answer_ids)
                    mapping.append((example_idx, branch_idx, "cf", cf_idx, alias_idx))

    log_probs: list[float] = []
    max_pseudo_batch = max(int(config.get("igsd_max_pseudo_batch", 256) or 256), 1)
    for start in range(0, len(prompts), max_pseudo_batch):
        chunk_log_probs, _ = _call_compute_log_prob(
            prompts[start : start + max_pseudo_batch],
            answers[start : start + max_pseudo_batch],
            tokenizer,
            actor_rollout_wg,
            log_prob_micro_batch_size_per_gpu,
            sanitize_nonfinite=False,
        )
        log_probs.extend(chunk_log_probs)
    score_lists: dict[tuple[int, int, str, int], dict[int, list[float]]] = {}
    for key, log_prob in zip(mapping, log_probs, strict=True):
        value = float(log_prob)
        if not np.isfinite(value):
            nonfinite_log_prob_count += 1
            continue
        example_idx, branch_idx, kind, cf_idx, alias_idx = key
        score_lists.setdefault((example_idx, branch_idx, kind, cf_idx), {}).setdefault(
            alias_idx, []
        ).append(value)
    scores: dict[tuple[int, int, str, int], dict[int, float]] = {
        key: {alias_idx: max(values) for alias_idx, values in aliases.items() if values}
        for key, aliases in score_lists.items()
    }

    valid_branch_count = 0
    valid_token_count = 0
    positive_gate_token_count = 0
    all_invalid_count = 0
    all_zero_gate_count = 0
    effective_token_count = 0
    fallback_token_count = 0
    all_gains: list[float] = []
    all_gates: list[float] = []
    all_common_cf_counts: list[int] = []
    all_common_alias_counts: list[int] = []
    all_common_alias_full_coverage: list[bool] = []
    disjoint_cf_token_count = 0
    alias_disjoint_cf_count = 0
    no_usable_paired_cf_token_count = 0
    nonfinite_pair_delta_count = 0
    row_gate_metrics: list[dict[str, float]] = []
    sampled_identity_stats: dict[str, dict[str, float]] = {}
    sampled_query_mean_gains: list[float] = []
    sampled_query_shared_gates: list[float] = []
    sampled_query_valid_pair_counts: list[int] = []
    sampled_query_active_pair_counts: list[int] = []
    sampled_funnel_stats = {
        "selected": 0.0,
        "record_aligned": 0.0,
        "both_parsed": 0.0,
        "both_retrieval_nonempty": 0.0,
        "scoreable": 0.0,
        "pair_ig_valid": 0.0,
        "positive_gain": 0.0,
        "exact_same_query": 0.0,
    }
    sampled_identity_mean_fields = (
        "pair_log_odds_shift",
        "student_top1_log_prob",
        "student_top1_prob",
        "teacher_pair_log_mass",
        "teacher_pair_mass",
        "student_pair_log_mass",
        "student_pair_mass",
        "teacher_pair_teacher_token_prob",
        "student_pair_teacher_token_prob",
        "prefilter_pair_jsd",
        "prefilter_pair_jsd_student_logit_grad",
        "prefilter_pair_jsd_student_logit_grad_abs",
        "teacher_top1_minus_reference_log_prob",
        "student_top1_minus_reference_log_prob",
        "teacher_top1_minus_sampled_log_prob",
        "student_top1_minus_sampled_log_prob",
        "student_top1_minus_teacher_log_prob",
    )

    def identity_stats(identity_state: str) -> dict[str, float]:
        state = identity_state or "unknown"
        return sampled_identity_stats.setdefault(
            state,
            {
                "count": 0.0,
                "selected": 0.0,
                "record_aligned": 0.0,
                "both_parsed": 0.0,
                "both_retrieval_nonempty": 0.0,
                "scoreable": 0.0,
                "exact_same_query": 0.0,
                "pair_ig_valid": 0.0,
                "positive_gain": 0.0,
                "gain_sum": 0.0,
                "positive_gain_sum": 0.0,
                "gate_sum": 0.0,
            },
        )

    metrics: dict[str, float] = {
        "igsd/token_intervention_scorer_example_count": float(len(examples)),
        "igsd/token_intervention_pseudo_sequence_count": float(len(prompts)),
        "igsd/token_intervention_forward_chunk_count": float(
            (len(prompts) + max_pseudo_batch - 1) // max_pseudo_batch
        ),
        "igsd/token_intervention_skipped_example_count": float(skipped_examples),
        "igsd/token_intervention_skipped_long_pseudo_count": float(skipped_long_pseudo),
        "igsd/token_intervention_skipped_empty_doc_pool_count": float(skipped_empty_doc_pool),
        "igsd/token_intervention_nonfinite_log_prob_count": float(nonfinite_log_prob_count),
        "igsd/token_intervention_doc_pool_size": float(len(doc_pool)),
        "igsd/sampled_pair_reference_mode_is_fixed_s": float(
            sampled_action_pairs and sampled_pair_reference_mode == "fixed_s"
        ),
    }

    if sampled_action_pairs:
        for ex in examples:
            token_count = sum(bool(value) for value in ex.get("student_action_query_mask", []))
            plans = ex.get("igsd_token_budget_plan", [])
            if len(plans) not in {0, token_count}:
                raise ValueError(
                    "Sampled-action pair plan must align with sampled query tokens before scoring, got "
                    f"{len(plans)=} and {token_count=}"
                )
            if not plans and token_count:
                plans = [{} for _ in range(token_count)]
                ex["igsd_token_budget_plan"] = plans
            records_by_index = {
                int(record["branch_index"]): record
                for record in ex.get("igsd_token_branch_records", [])
            }
            for plan in plans:
                identity = identity_stats(str(plan.get("identity_state", "unknown")))
                identity["count"] += 1.0
                for field in sampled_identity_mean_fields:
                    value = plan.get(field)
                    if value is None or not np.isfinite(float(value)):
                        continue
                    identity[f"{field}_sum"] = identity.get(f"{field}_sum", 0.0) + float(value)
                    identity[f"{field}_count"] = identity.get(f"{field}_count", 0.0) + 1.0

                selected = bool(plan.get("selected", False))
                identity["selected"] += float(selected)
                plan["pair_scoreable"] = False
                plan["pair_ig_error"] = None
                if not selected:
                    continue
                sampled_funnel_stats["selected"] += 1.0
                teacher_branch_idx = plan.get("teacher_branch_index")
                sampled_branch_idx = plan.get("sampled_branch_index")
                teacher_record = (
                    records_by_index.get(int(teacher_branch_idx))
                    if teacher_branch_idx is not None
                    else None
                )
                sampled_record = (
                    records_by_index.get(int(sampled_branch_idx))
                    if sampled_branch_idx is not None
                    else None
                )
                expected_reference_kind = (
                    "student_terminal"
                    if sampled_pair_reference_mode == "fixed_s"
                    else "sampled_pair_student"
                )

                def optional_id_matches(record: dict[str, Any] | None, key: str, expected: Any) -> bool:
                    if record is None or record.get(key) is None or expected is None:
                        return record is not None
                    return int(record[key]) == int(expected)

                record_aligned = bool(
                    teacher_record is not None
                    and sampled_record is not None
                    and teacher_record.get("branch_kind") == "sampled_pair_teacher"
                    and sampled_record.get("branch_kind") == expected_reference_kind
                    and optional_id_matches(
                        teacher_record, "candidate_token_id", plan.get("teacher_top1_id")
                    )
                    and (
                        sampled_pair_reference_mode == "fixed_s"
                        or optional_id_matches(
                            sampled_record,
                            "candidate_token_id",
                            plan.get("reference_token_id", plan.get("sampled_token_id")),
                        )
                    )
                )
                plan["pair_record_alignment_valid"] = record_aligned
                identity["record_aligned"] += float(record_aligned)
                sampled_funnel_stats["record_aligned"] += float(record_aligned)

                teacher_parsed = bool(
                    teacher_record is not None
                    and teacher_record.get("branch_valid", False)
                    and str(teacher_record.get("branch_query", "")).strip()
                )
                sampled_parsed = bool(
                    sampled_record is not None
                    and sampled_record.get("branch_valid", False)
                    and str(sampled_record.get("branch_query", "")).strip()
                )
                both_parsed = teacher_parsed and sampled_parsed
                plan["pair_both_parsed"] = both_parsed
                identity["both_parsed"] += float(both_parsed)
                sampled_funnel_stats["both_parsed"] += float(both_parsed)

                both_retrieval_nonempty = bool(
                    both_parsed
                    and _dedup_docs(teacher_record.get("branch_documents", []))
                    and _dedup_docs(sampled_record.get("branch_documents", []))
                )
                plan["pair_both_retrieval_nonempty"] = both_retrieval_nonempty
                identity["both_retrieval_nonempty"] += float(both_retrieval_nonempty)
                sampled_funnel_stats["both_retrieval_nonempty"] += float(
                    both_retrieval_nonempty
                )

                exact_same_query = bool(
                    both_parsed
                    and (
                        [str(query).strip() for query in teacher_record.get("branch_queries", [])]
                        or [str(teacher_record.get("branch_query", "")).strip()]
                    )
                    == (
                        [str(query).strip() for query in sampled_record.get("branch_queries", [])]
                        or [str(sampled_record.get("branch_query", "")).strip()]
                    )
                )
                plan["pair_exact_same_query"] = exact_same_query
                identity["exact_same_query"] += float(exact_same_query)
                sampled_funnel_stats["exact_same_query"] += float(exact_same_query)
                if not record_aligned:
                    plan["pair_ig_error"] = "branch_record_alignment_mismatch"
                elif not both_parsed:
                    plan["pair_ig_error"] = "branch_parse_invalid"
                elif not both_retrieval_nonempty:
                    plan["pair_ig_error"] = "empty_branch_retrieval"

    for example_idx, ex in enumerate(examples):
        records = ex.get("igsd_token_branch_records", [])
        if not records:
            token_count = sum(bool(value) for value in ex.get("student_action_query_mask", []))
            gains = np.zeros(token_count, dtype=np.float32)
            token_valid = np.zeros(token_count, dtype=bool)
            gates, gate_metrics = _token_gate_and_row_normalize(gains, token_valid, config)
            if sampled_action_pairs:
                plans = ex.get("igsd_token_budget_plan", [])
                if len(plans) not in {0, token_count}:
                    raise ValueError(
                        "Sampled-action pair plan must align with sampled query tokens for an "
                        "unscored row, got "
                        f"{len(plans)=} and {token_count=}"
                    )
                if not plans and token_count:
                    # A row with no continuation requests has no eligible
                    # supervision. Preserve its query denominator explicitly
                    # so the later target packer can remain position-aligned.
                    plans = [{} for _ in range(token_count)]
                    ex["igsd_token_budget_plan"] = plans
                # This mode has no unverified fallback path. Enforce that
                # invariant locally even if a direct caller bypasses config
                # validation and supplies a non-zero fallback weight.
                for plan in plans:
                    plan["pair_gate_source"] = "environment_ig"
                    plan["pair_ig_valid"] = False
                    plan["pair_gate_valid"] = False
                    plan["pair_scoreable"] = False
                    plan["pair_gate"] = 0.0
                    plan["pair_positive_gain"] = False
                    if bool(plan.get("selected", False)) and plan.get("pair_ig_error") is None:
                        plan["pair_ig_error"] = "missing_branch_records"
                    plan.pop("pair_left_ig", None)
                    plan.pop("pair_right_ig", None)
                    plan.pop("pair_gain", None)
                gates.fill(0.0)
                effective_valid = np.ones(token_count, dtype=bool)
                gate_metrics.update(
                    {
                        "igsd/token_intervention_normalized_gate_mean": 0.0,
                        "igsd/token_intervention_normalized_gate_max": 0.0,
                        "igsd/token_intervention_output_gate_mean": 0.0,
                        "igsd/token_intervention_output_gate_max": 0.0,
                        "igsd/token_intervention_effective_token_count": 0.0,
                        "igsd/token_intervention_fallback_token_count": 0.0,
                        "igsd/token_intervention_fallback_token_frac": 0.0,
                        "igsd/token_intervention_positive_gate_frac": 0.0,
                        "igsd/token_intervention_gate_zero_frac": (
                            1.0 if token_count else 0.0
                        ),
                    }
                )
            else:
                effective_valid = token_valid | (gates > 0.0)
            row_gate_metrics.append(gate_metrics)
            effective_token_count += int(effective_valid.sum())
            if not sampled_action_pairs:
                fallback_token_count += int((effective_valid & ~token_valid).sum())
            all_gates.extend(gates[effective_valid].astype(float).tolist())
            if token_count > 0:
                all_invalid_count += 1
                if not np.any(gates[effective_valid] > 0.0):
                    all_zero_gate_count += 1
            ex["igsd_token_ig_values"] = []
            ex["igsd_token_branch_ig_valid"] = []
            ex["igsd_token_gains"] = gains.tolist()
            ex["igsd_token_gate_signals"] = gains.tolist()
            ex["igsd_token_gate"] = gates.tolist()
            ex["igsd_token_delta_valid"] = token_valid.tolist()
            ex["igsd_token_gate_valid"] = effective_valid.tolist()
            ex["igsd_token_pair_left_ig"] = gains.tolist()
            ex["igsd_token_pair_right_ig"] = gains.tolist()
            ex["igsd_token_common_cf_count"] = [0] * token_count
            ex["igsd_token_common_alias_count_min"] = [0] * token_count
            ex["igsd_token_common_alias_count_mean"] = gains.tolist()
            continue
        branch_count = len(records)
        ig_values = np.zeros(branch_count, dtype=np.float32)
        branch_ig_valid = np.zeros(branch_count, dtype=bool)
        branch_real_scores: dict[int, dict[int, float]] = {}
        branch_cf_scores: dict[int, dict[int, dict[int, float]]] = {}
        for record in records:
            record["branch_ig_valid"] = False
            branch_idx = int(record["branch_index"])
            real_alias_scores = dict(
                scores.get((example_idx, branch_idx, "real", 0), {})
            )
            cf_alias_scores = {
                cf_idx: dict(scores.get((example_idx, branch_idx, "cf", cf_idx), {}))
                for cf_idx in sorted(
                    attempted_cf_by_branch.get((example_idx, branch_idx), set())
                )
            }
            if (example_idx, branch_idx) in attempted_real_branches:
                branch_real_scores[branch_idx] = real_alias_scores
            if cf_alias_scores:
                branch_cf_scores[branch_idx] = cf_alias_scores

            # Branch IG remains a best-available diagnostic for compatibility.
            # Training gains below are recomputed from four-cell matched alias sets.
            real_values = list(real_alias_scores.values())
            cf_values = {
                cf_idx: max(alias_scores.values())
                for cf_idx, alias_scores in cf_alias_scores.items()
                if alias_scores
            }
            paired_cf = sorted(cf_values)
            if not real_values or not paired_cf:
                skipped_branches += 1
                continue
            real_score = float(max(real_values))
            branch_ig = float(real_score - np.mean([cf_values[idx] for idx in paired_cf]))
            if not np.isfinite(branch_ig):
                record["branch_error"] = "non_finite_branch_ig"
                skipped_branches += 1
                continue
            ig_values[branch_idx] = branch_ig
            branch_ig_valid[branch_idx] = True
            record["branch_ig"] = float(ig_values[branch_idx])
            record["branch_ig_valid"] = True
            record["branch_ig_alias_mode"] = "best_available"
            valid_branch_count += 1

        if sampled_action_pairs:
            query_token_count = sum(bool(value) for value in ex.get("student_action_query_mask", []))
            plans = ex.get("igsd_token_budget_plan", [])
            if len(plans) not in {0, query_token_count}:
                raise ValueError(
                    "Sampled-action pair plan must align with sampled query tokens while scoring, got "
                    f"{len(plans)=} and {query_token_count=}"
                )
            if not plans and query_token_count:
                plans = [{} for _ in range(query_token_count)]
                ex["igsd_token_budget_plan"] = plans
            token_count = len(plans)
            gains = np.zeros(token_count, dtype=np.float32)
            gate_signals = np.zeros(token_count, dtype=np.float32)
            token_valid = np.zeros(token_count, dtype=bool)
            pair_left_ig = np.zeros(token_count, dtype=np.float32)
            pair_right_ig = np.zeros(token_count, dtype=np.float32)
            common_cf_count = np.zeros(token_count, dtype=np.int32)
            common_alias_count_min = np.zeros(token_count, dtype=np.int32)
            common_alias_count_mean = np.zeros(token_count, dtype=np.float32)
            utility_margin = float(config.get("igsd_token_gate_margin", 0.0))
            records_by_index = {
                int(record["branch_index"]): record for record in records
            }

            for token_idx, plan in enumerate(plans):
                # Routing decisions are final: consensus, direction-rejected,
                # invalid, and budget-skipped positions were never assigned a
                # matched pair and therefore are not scoring failures.
                plan["pair_gate_source"] = "environment_ig"
                plan["pair_gate_granularity"] = sampled_pair_gate_granularity
                plan["pair_ig_valid"] = False
                plan["pair_gate_valid"] = False
                plan["pair_scoreable"] = False
                plan.pop("pair_left_ig", None)
                plan.pop("pair_right_ig", None)
                plan.pop("pair_gain", None)
                teacher_branch_idx = plan.get("teacher_branch_index")
                sampled_branch_idx = plan.get("sampled_branch_index")
                plan["teacher_branch_valid"] = bool(
                    teacher_branch_idx is not None
                    and records_by_index.get(int(teacher_branch_idx), {}).get("branch_valid", False)
                )
                plan["sampled_branch_valid"] = bool(
                    sampled_branch_idx is not None
                    and records_by_index.get(int(sampled_branch_idx), {}).get("branch_valid", False)
                )
                if not bool(plan.get("selected", False)):
                    continue
                teacher_real = (
                    branch_real_scores.get(int(teacher_branch_idx))
                    if teacher_branch_idx is not None
                    else None
                )
                sampled_real = (
                    branch_real_scores.get(int(sampled_branch_idx))
                    if sampled_branch_idx is not None
                    else None
                )
                teacher_cf = (
                    branch_cf_scores.get(int(teacher_branch_idx))
                    if teacher_branch_idx is not None
                    else None
                )
                sampled_cf = (
                    branch_cf_scores.get(int(sampled_branch_idx))
                    if sampled_branch_idx is not None
                    else None
                )
                scoreable = bool(
                    teacher_real
                    and sampled_real
                    and teacher_cf
                    and sampled_cf
                    and any(bool(values) for values in teacher_cf.values())
                    and any(bool(values) for values in sampled_cf.values())
                )
                plan["pair_scoreable"] = scoreable
                identity = identity_stats(str(plan.get("identity_state", "unknown")))
                identity["scoreable"] += float(scoreable)
                sampled_funnel_stats["scoreable"] += float(scoreable)
                stats, reason = _matched_branch_pair_ig(
                    teacher_real,
                    sampled_real,
                    teacher_cf,
                    sampled_cf,
                    alias_counts[example_idx],
                )
                if stats is None:
                    if reason == "disjoint_counterfactuals":
                        disjoint_cf_token_count += 1
                    elif reason == "no_common_aliases":
                        alias_disjoint_cf_count += 1
                    elif reason == "nonfinite_pair_delta":
                        nonfinite_pair_delta_count += 1
                    else:
                        no_usable_paired_cf_token_count += 1
                    plan["pair_ig_valid"] = False
                    if plan.get("pair_ig_error") is None:
                        plan["pair_ig_error"] = (
                            "missing_answer_aliases"
                            if alias_counts[example_idx] <= 0
                            else str(reason or "missing_branch_scores")
                        )
                    continue
                gain = float(stats["gain"])
                gains[token_idx] = gain
                token_valid[token_idx] = True
                pair_left_ig[token_idx] = float(stats["left_ig"])
                pair_right_ig[token_idx] = float(stats["right_ig"])
                common_cf_count[token_idx] = int(stats["common_cf_count"])
                common_alias_count_min[token_idx] = int(stats["common_alias_count_min"])
                common_alias_count_mean[token_idx] = float(stats["common_alias_count_mean"])
                plan["pair_ig_valid"] = True
                plan["pair_gate_valid"] = True
                plan["pair_ig_error"] = None
                plan["pair_left_ig"] = float(stats["left_ig"])
                plan["pair_right_ig"] = float(stats["right_ig"])
                plan["pair_gain"] = gain
                plan["pair_gate_signal"] = gain
                all_common_alias_counts.extend(stats["common_alias_counts"])
                all_common_alias_full_coverage.extend(stats["common_alias_full_coverage"])
                identity["pair_ig_valid"] += 1.0
                identity["gain_sum"] += gain
                sampled_funnel_stats["pair_ig_valid"] += 1.0

            token_valid &= np.isfinite(gains)
            (
                gates,
                gate_signals,
                gate_metrics,
                query_mean_gain,
                query_shared_gate,
            ) = _sampled_pair_gate_from_gains(gains, token_valid, config)
            # A sampled pair is a strict verified correction. Invalid pairs,
            # non-positive utility gains, and budget skips must not fall back
            # to ordinary OPD or an unverified consensus loss.
            gates[~(token_valid & (gate_signals > utility_margin))] = 0.0
            # Every query token remains in the denominator, including skipped
            # and invalid pairs. This keeps sparse acceptance from becoming an
            # implicit row-level normalization.
            denominator_valid = np.ones(token_count, dtype=bool)
            gate_metrics["igsd/token_intervention_valid_token_count"] = float(token_valid.sum())
            gate_metrics["igsd/token_intervention_valid_token_frac"] = float(
                token_valid.sum() / max(token_count, 1)
            )
            gate_metrics["igsd/token_intervention_output_gate_mean"] = float(
                gates.mean() if gates.size else 0.0
            )
            gate_metrics["igsd/token_intervention_output_gate_max"] = float(
                gates.max() if gates.size else 0.0
            )
            gate_metrics["igsd/token_intervention_effective_token_count"] = float(
                (gates > 0.0).sum()
            )
            gate_metrics["igsd/token_intervention_fallback_token_count"] = 0.0
            gate_metrics["igsd/token_intervention_fallback_token_frac"] = 0.0
            gate_metrics["igsd/token_intervention_normalized_gate_mean"] = float(
                gates.mean() if gates.size else 0.0
            )
            gate_metrics["igsd/token_intervention_normalized_gate_max"] = float(
                gates.max() if gates.size else 0.0
            )
            gate_metrics["igsd/token_intervention_positive_gate_frac"] = float(
                np.mean(gates > 0.0) if gates.size else 0.0
            )
            gate_metrics["igsd/token_intervention_gate_zero_frac"] = float(
                np.mean(gates <= 0.0) if gates.size else 0.0
            )
            gate_metrics["igsd/sampled_pair_gate_granularity_is_token"] = float(
                sampled_pair_gate_granularity == "token"
            )
            gate_metrics["igsd/sampled_pair_gate_granularity_is_query_mean"] = float(
                sampled_pair_gate_granularity == "query_mean"
            )
            gate_metrics["igsd/sampled_pair_query_mean_gain"] = query_mean_gain
            gate_metrics["igsd/sampled_pair_query_shared_gate"] = query_shared_gate
            row_gate_metrics.append(gate_metrics)
            sampled_query_mean_gains.append(query_mean_gain)
            sampled_query_shared_gates.append(query_shared_gate)
            sampled_query_valid_pair_counts.append(int(token_valid.sum()))
            sampled_query_active_pair_counts.append(int(np.sum(gates > 0.0)))
            effective_token_count += int(denominator_valid.sum())
            if token_valid.any():
                valid_token_count += int(token_valid.sum())
                all_gains.extend(gains[token_valid].astype(float).tolist())
                all_common_cf_counts.extend(common_cf_count[token_valid].astype(int).tolist())
            elif token_count > 0:
                all_invalid_count += 1
            positive_gate_token_count += int(np.sum(gates > 0.0))
            all_gates.extend(gates.astype(float).tolist())
            if token_count > 0 and not np.any(gates > 0.0):
                all_zero_gate_count += 1
            ex["igsd_token_ig_values"] = ig_values.tolist()
            ex["igsd_token_branch_ig_valid"] = branch_ig_valid.tolist()
            ex["igsd_token_gains"] = gains.tolist()
            ex["igsd_token_gate_signals"] = gate_signals.tolist()
            ex["igsd_token_gate"] = gates.tolist()
            ex["igsd_token_delta_valid"] = token_valid.tolist()
            ex["igsd_token_gate_valid"] = denominator_valid.tolist()
            ex["igsd_token_pair_left_ig"] = pair_left_ig.tolist()
            ex["igsd_token_pair_right_ig"] = pair_right_ig.tolist()
            ex["igsd_token_common_cf_count"] = common_cf_count.tolist()
            ex["igsd_token_common_alias_count_min"] = common_alias_count_min.tolist()
            ex["igsd_token_common_alias_count_mean"] = common_alias_count_mean.tolist()
            ex["igsd_sampled_pair_query_mean_gain"] = float(query_mean_gain)
            ex["igsd_sampled_pair_query_shared_gate"] = float(query_shared_gate)
            ex["igsd_sampled_pair_query_valid_pair_count"] = int(token_valid.sum())
            ex["igsd_sampled_pair_query_active_pair_count"] = int(np.sum(gates > 0.0))
            for token_idx, plan in enumerate(ex.get("igsd_token_budget_plan", [])):
                stats = identity_stats(str(plan.get("identity_state", "unknown")))
                is_valid = bool(token_valid[token_idx])
                gain = float(gains[token_idx])
                gate_signal = float(gate_signals[token_idx])
                plan["pair_gate_granularity"] = sampled_pair_gate_granularity
                plan["pair_gate_signal"] = gate_signal
                positive = bool(plan.get("selected", False)) and (
                    is_valid
                    and gate_signal > utility_margin
                    and gates[token_idx] > 0.0
                )
                plan["pair_gate"] = float(gates[token_idx])
                plan["pair_active"] = bool(positive)
                plan["pair_positive_gain"] = positive
                stats["positive_gain"] += float(positive)
                if positive:
                    stats["positive_gain_sum"] += gain
                    stats["gate_sum"] += float(gates[token_idx])
                    sampled_funnel_stats["positive_gain"] += 1.0
            continue

        if budgeted_local_candidates:
            plans = ex["igsd_token_budget_plan"]
            token_count = len(plans)
            gains = np.zeros(token_count, dtype=np.float32)
            token_valid = np.zeros(token_count, dtype=bool)
            confirmation_mask = np.zeros(token_count, dtype=bool)
            confirmation_gates = np.zeros(token_count, dtype=np.float32)
            pair_left_ig = np.zeros(token_count, dtype=np.float32)
            pair_right_ig = np.zeros(token_count, dtype=np.float32)
            common_cf_count = np.zeros(token_count, dtype=np.int32)
            common_alias_count_min = np.zeros(token_count, dtype=np.int32)
            common_alias_count_mean = np.zeros(token_count, dtype=np.float32)
            terminal_records = [
                record for record in records if record.get("branch_kind") == "student_terminal"
            ]
            if len(terminal_records) != 1:
                raise ValueError(
                    "Budgeted local candidate scoring requires exactly one shared student terminal branch, "
                    f"got {len(terminal_records)}"
                )
            terminal_idx = int(terminal_records[0]["branch_index"])
            target_tilts: list[dict[str, Any]] = []
            confirmation_log_gaps = np.zeros(token_count, dtype=np.float32)
            utility_margin = float(config.get("igsd_token_gate_margin", 0.0))

            for token_idx, plan in enumerate(plans):
                state = str(plan.get("state", "invalid_prefilter"))
                if state == "confirmation":
                    confirmation_mask[token_idx] = True
                    confirmation_gates[token_idx] = float(plan.get("confirmation_gate", 0.0))
                    confirmation_log_gaps[token_idx] = float(
                        plan.get("teacher_student_log_gap", 0.0)
                    )
                    continue
                candidate_stats: list[dict[str, Any]] = []
                for rank in (1, 2):
                    branch_idx = plan.get(f"top{rank}_branch_index")
                    if branch_idx is None:
                        continue
                    branch_idx = int(branch_idx)
                    stats, reason = _matched_branch_pair_ig(
                        branch_real_scores.get(branch_idx),
                        branch_real_scores.get(terminal_idx),
                        branch_cf_scores.get(branch_idx),
                        branch_cf_scores.get(terminal_idx),
                        alias_counts[example_idx],
                    )
                    if stats is None:
                        if reason == "disjoint_counterfactuals":
                            disjoint_cf_token_count += 1
                        elif reason == "no_common_aliases":
                            alias_disjoint_cf_count += 1
                        elif reason == "nonfinite_pair_delta":
                            nonfinite_pair_delta_count += 1
                        else:
                            no_usable_paired_cf_token_count += 1
                        continue
                    stats["rank"] = rank
                    stats["branch_index"] = branch_idx
                    candidate_stats.append(stats)
                    all_common_alias_counts.extend(stats["common_alias_counts"])
                    all_common_alias_full_coverage.extend(stats["common_alias_full_coverage"])

                if not candidate_stats:
                    continue
                best = max(candidate_stats, key=lambda item: float(item["gain"]))
                gains[token_idx] = float(best["gain"])
                token_valid[token_idx] = True
                pair_left_ig[token_idx] = float(best["left_ig"])
                pair_right_ig[token_idx] = float(best["right_ig"])
                common_cf_count[token_idx] = int(best["common_cf_count"])
                common_alias_count_min[token_idx] = int(best["common_alias_count_min"])
                common_alias_count_mean[token_idx] = float(best["common_alias_count_mean"])
                top1_stats = next((item for item in candidate_stats if item["rank"] == 1), None)
                top2_stats = next((item for item in candidate_stats if item["rank"] == 2), None)
                if (
                    top1_stats is not None
                    and top2_stats is not None
                    and float(best["gain"]) > utility_margin
                ):
                    target_tilts.append(
                        {
                            "action_query_position": int(plan["action_query_position"]),
                            "teacher_top1_id": int(plan["teacher_top1_id"]),
                            "teacher_top2_id": int(plan["teacher_top2_id"]),
                            "top1_ig": float(top1_stats["left_ig"]),
                            "top2_ig": float(top2_stats["left_ig"]),
                        }
                    )

            utility_gates, gate_metrics = _token_gate_and_row_normalize(gains, token_valid, config)
            # Unlike the legacy sigmoid G1 variant, a local candidate that
            # loses to the shared student terminal must not receive a positive
            # auxiliary weight merely because sigmoid is nonzero everywhere.
            utility_gates[~(token_valid & (gains > utility_margin))] = 0.0
            gates = utility_gates.copy()
            gates[confirmation_mask] = confirmation_gates[confirmation_mask]
            effective_valid = token_valid | confirmation_mask
            effective_gate_values = gates[effective_valid]
            gate_metrics["igsd/token_intervention_valid_token_count"] = float(token_valid.sum())
            gate_metrics["igsd/token_intervention_valid_token_frac"] = float(
                token_valid.sum() / max(token_count, 1)
            )
            gate_metrics["igsd/token_intervention_output_gate_mean"] = float(
                effective_gate_values.mean() if effective_gate_values.size else 0.0
            )
            gate_metrics["igsd/token_intervention_output_gate_max"] = float(
                effective_gate_values.max() if effective_gate_values.size else 0.0
            )
            gate_metrics["igsd/token_intervention_effective_token_count"] = float(effective_valid.sum())
            gate_metrics["igsd/token_intervention_fallback_token_count"] = 0.0
            row_gate_metrics.append(gate_metrics)
            effective_token_count += int(effective_valid.sum())
            if token_valid.any():
                valid_token_count += int(token_valid.sum())
                all_gains.extend(gains[token_valid].astype(float).tolist())
                all_common_cf_counts.extend(common_cf_count[token_valid].astype(int).tolist())
            elif token_count > 0 and not confirmation_mask.any():
                all_invalid_count += 1
            positive_gate_token_count += int(np.sum(gates[effective_valid] > 0.0))
            all_gates.extend(gates[effective_valid].astype(float).tolist())
            if token_count > 0 and not np.any(gates[effective_valid] > 0.0):
                all_zero_gate_count += 1
            ex["igsd_token_ig_values"] = ig_values.tolist()
            ex["igsd_token_branch_ig_valid"] = branch_ig_valid.tolist()
            ex["igsd_token_gains"] = gains.tolist()
            ex["igsd_token_gate"] = gates.tolist()
            ex["igsd_token_delta_valid"] = token_valid.tolist()
            ex["igsd_token_gate_valid"] = effective_valid.tolist()
            ex["igsd_token_pair_left_ig"] = pair_left_ig.tolist()
            ex["igsd_token_pair_right_ig"] = pair_right_ig.tolist()
            ex["igsd_token_common_cf_count"] = common_cf_count.tolist()
            ex["igsd_token_common_alias_count_min"] = common_alias_count_min.tolist()
            ex["igsd_token_common_alias_count_mean"] = common_alias_count_mean.tolist()
            ex["igsd_budgeted_confirmation_log_gap"] = confirmation_log_gaps.tolist()
            ex["igsd_budgeted_target_tilts"] = target_tilts
            continue

        token_count = max(branch_count - 1, 0)
        gains = np.zeros(token_count, dtype=np.float32)
        token_valid = np.zeros(token_count, dtype=bool)
        pair_left_ig = np.zeros(token_count, dtype=np.float32)
        pair_right_ig = np.zeros(token_count, dtype=np.float32)
        common_cf_count = np.zeros(token_count, dtype=np.int32)
        common_alias_count_min = np.zeros(token_count, dtype=np.int32)
        common_alias_count_mean = np.zeros(token_count, dtype=np.float32)
        for token_idx in range(token_count):
            left_real = branch_real_scores.get(token_idx)
            right_real = branch_real_scores.get(token_idx + 1)
            left_cf = branch_cf_scores.get(token_idx)
            right_cf = branch_cf_scores.get(token_idx + 1)
            if left_real is None or right_real is None or left_cf is None or right_cf is None:
                no_usable_paired_cf_token_count += 1
                continue
            shared_cf = sorted(left_cf.keys() & right_cf.keys())
            if not shared_cf:
                disjoint_cf_token_count += 1
                no_usable_paired_cf_token_count += 1
                continue
            left_deltas: list[float] = []
            right_deltas: list[float] = []
            common_alias_counts: list[int] = []
            common_alias_full_coverage: list[bool] = []
            for cf_idx in shared_cf:
                common_aliases = sorted(
                    left_real.keys()
                    & right_real.keys()
                    & left_cf[cf_idx].keys()
                    & right_cf[cf_idx].keys()
                )
                if not common_aliases:
                    alias_disjoint_cf_count += 1
                    continue
                left_delta = float(
                    max(left_real[alias_idx] for alias_idx in common_aliases)
                    - max(left_cf[cf_idx][alias_idx] for alias_idx in common_aliases)
                )
                right_delta = float(
                    max(right_real[alias_idx] for alias_idx in common_aliases)
                    - max(right_cf[cf_idx][alias_idx] for alias_idx in common_aliases)
                )
                if not np.isfinite(left_delta) or not np.isfinite(right_delta):
                    nonfinite_pair_delta_count += 1
                    continue
                left_deltas.append(left_delta)
                right_deltas.append(right_delta)
                common_alias_counts.append(len(common_aliases))
                common_alias_full_coverage.append(
                    len(common_aliases) == alias_counts[example_idx]
                )
            if not left_deltas:
                no_usable_paired_cf_token_count += 1
                continue
            left_ig = float(np.mean(left_deltas))
            right_ig = float(np.mean(right_deltas))
            gain = float(np.mean(np.asarray(left_deltas) - np.asarray(right_deltas)))
            if not np.isfinite(left_ig) or not np.isfinite(right_ig) or not np.isfinite(gain):
                nonfinite_pair_delta_count += 1
                no_usable_paired_cf_token_count += 1
                continue
            pair_left_ig[token_idx] = left_ig
            pair_right_ig[token_idx] = right_ig
            common_cf_count[token_idx] = len(left_deltas)
            common_alias_count_min[token_idx] = min(common_alias_counts)
            common_alias_count_mean[token_idx] = float(np.mean(common_alias_counts))
            all_common_alias_counts.extend(common_alias_counts)
            all_common_alias_full_coverage.extend(common_alias_full_coverage)
            gains[token_idx] = gain
            token_valid[token_idx] = True
        token_valid &= np.isfinite(gains)
        gates, gate_metrics = _token_gate_and_row_normalize(gains, token_valid, config)
        effective_valid = token_valid | (gates > 0.0)
        row_gate_metrics.append(gate_metrics)
        effective_token_count += int(effective_valid.sum())
        fallback_token_count += int((effective_valid & ~token_valid).sum())
        if token_valid.any():
            valid_token_count += int(token_valid.sum())
            positive_gate_token_count += int(np.sum(gates[token_valid] > 0.0))
            all_gains.extend(gains[token_valid].astype(float).tolist())
            all_common_cf_counts.extend(common_cf_count[token_valid].astype(int).tolist())
        elif token_count > 0:
            all_invalid_count += 1
        all_gates.extend(gates[effective_valid].astype(float).tolist())
        if token_count > 0 and not np.any(gates[effective_valid] > 0.0):
            all_zero_gate_count += 1
        ex["igsd_token_ig_values"] = ig_values.tolist()
        ex["igsd_token_branch_ig_valid"] = branch_ig_valid.tolist()
        ex["igsd_token_gains"] = gains.tolist()
        ex["igsd_token_gate_signals"] = gains.tolist()
        ex["igsd_token_gate"] = gates.tolist()
        ex["igsd_token_delta_valid"] = token_valid.tolist()
        ex["igsd_token_gate_valid"] = effective_valid.tolist()
        ex["igsd_token_pair_left_ig"] = pair_left_ig.tolist()
        ex["igsd_token_pair_right_ig"] = pair_right_ig.tolist()
        ex["igsd_token_common_cf_count"] = common_cf_count.tolist()
        ex["igsd_token_common_alias_count_min"] = common_alias_count_min.tolist()
        ex["igsd_token_common_alias_count_mean"] = common_alias_count_mean.tolist()

    for key in (
        "igsd/token_intervention_raw_gate_mean",
        "igsd/token_intervention_raw_gate_max",
        "igsd/token_intervention_gate_zero_frac",
        "igsd/token_intervention_positive_gate_frac",
    ):
        metrics[key] = float(np.mean([row[key] for row in row_gate_metrics])) if row_gate_metrics else 0.0
    metrics.update(
        {
            "igsd/token_intervention_valid_branch_count": float(valid_branch_count),
            "igsd/token_intervention_valid_branch_frac": float(
                valid_branch_count / max(sum(branch_counts), 1)
            ),
            "igsd/token_intervention_valid_token_count": float(valid_token_count),
            "igsd/token_intervention_valid_token_frac": float(
                valid_token_count
                / max(
                    sum(query_token_counts)
                    if budgeted_local_candidates or sampled_action_pairs
                    else sum(max(count - 1, 0) for count in branch_counts),
                    1,
                )
            ),
            "igsd/token_intervention_effective_token_count": float(effective_token_count),
            "igsd/token_intervention_fallback_token_count": float(fallback_token_count),
            "igsd/token_intervention_fallback_token_frac": float(
                fallback_token_count / max(effective_token_count, 1)
            ),
            "igsd/token_intervention_skipped_branch_count": float(skipped_branches),
            "igsd/token_intervention_all_invalid_query_count": float(all_invalid_count),
            "igsd/token_intervention_all_zero_gate_query_count": float(all_zero_gate_count),
            "igsd/token_intervention_positive_gate_token_count": float(positive_gate_token_count),
            "igsd/token_intervention_positive_gate_token_frac": float(
                positive_gate_token_count
                / max(effective_token_count if sampled_action_pairs else valid_token_count, 1)
            ),
            "igsd/token_intervention_disjoint_cf_token_count": float(
                disjoint_cf_token_count
            ),
            "igsd/token_intervention_alias_disjoint_cf_count": float(
                alias_disjoint_cf_count
            ),
            "igsd/token_intervention_no_usable_paired_cf_token_count": float(
                no_usable_paired_cf_token_count
            ),
            "igsd/token_intervention_nonfinite_pair_delta_count": float(
                nonfinite_pair_delta_count
            ),
            "igsd/token_intervention_common_cf_count_mean": float(
                np.mean(all_common_cf_counts)
            )
            if all_common_cf_counts
            else 0.0,
            "igsd/token_intervention_common_cf_count_min": float(
                min(all_common_cf_counts, default=0)
            ),
            "igsd/token_intervention_common_cf_count_max": float(
                max(all_common_cf_counts, default=0)
            ),
            "igsd/token_intervention_common_cf_full_coverage_frac": float(
                np.mean([count == num_cf for count in all_common_cf_counts])
            )
            if all_common_cf_counts
            else 0.0,
            "igsd/token_intervention_common_alias_count_mean": float(
                np.mean(all_common_alias_counts)
            )
            if all_common_alias_counts
            else 0.0,
            "igsd/token_intervention_common_alias_count_min": float(
                min(all_common_alias_counts, default=0)
            ),
            "igsd/token_intervention_common_alias_count_max": float(
                max(all_common_alias_counts, default=0)
            ),
            "igsd/token_intervention_common_alias_full_coverage_frac": float(
                np.mean(all_common_alias_full_coverage)
            )
            if all_common_alias_full_coverage
            else 0.0,
            "igsd/token_intervention_gain_mean": float(np.mean(all_gains)) if all_gains else 0.0,
            "igsd/token_intervention_gain_std": float(np.std(all_gains)) if all_gains else 0.0,
            "igsd/token_intervention_normalized_gate_mean": float(np.mean(all_gates)) if all_gates else 0.0,
            "igsd/token_intervention_normalized_gate_std": float(np.std(all_gates)) if all_gates else 0.0,
            "igsd/token_intervention_normalized_gate_max": float(max(all_gates)) if all_gates else 0.0,
            "igsd/token_intervention_output_gate_mean": float(np.mean(all_gates)) if all_gates else 0.0,
            "igsd/token_intervention_output_gate_std": float(np.std(all_gates)) if all_gates else 0.0,
            "igsd/token_intervention_output_gate_max": float(max(all_gates)) if all_gates else 0.0,
            "igsd/token_intervention_gate_normalization_is_row_mean": float(
                str(config.get("igsd_token_gate_normalization", "row_mean")).lower()
                == "row_mean"
            ),
            "igsd/token_intervention_gate_normalization_is_none": float(
                str(config.get("igsd_token_gate_normalization", "row_mean")).lower() == "none"
            ),
            "igsd/token_intervention_time_sec": time.perf_counter() - started,
            "igsd/sampled_pair_gate_granularity_is_token": float(
                sampled_pair_gate_granularity == "token"
            )
            if sampled_action_pairs
            else 0.0,
            "igsd/sampled_pair_gate_granularity_is_query_mean": float(
                sampled_pair_gate_granularity == "query_mean"
            )
            if sampled_action_pairs
            else 0.0,
            "igsd/sampled_pair_query_mean_gain_mean": float(
                np.mean(sampled_query_mean_gains)
            )
            if sampled_query_mean_gains
            else 0.0,
            "igsd/sampled_pair_query_shared_gate_mean": float(
                np.mean(sampled_query_shared_gates)
            )
            if sampled_query_shared_gates
            else 0.0,
            "igsd/sampled_pair_query_valid_pair_count_mean": float(
                np.mean(sampled_query_valid_pair_counts)
            )
            if sampled_query_valid_pair_counts
            else 0.0,
            "igsd/sampled_pair_query_active_pair_count_mean": float(
                np.mean(sampled_query_active_pair_counts)
            )
            if sampled_query_active_pair_counts
            else 0.0,
        }
    )
    if sampled_action_pairs:
        selected_count = sampled_funnel_stats["selected"]
        for stage, count in sampled_funnel_stats.items():
            metrics[f"igsd/sampled_pair_funnel_{stage}_count"] = count
            metrics[f"igsd/sampled_pair_funnel_{stage}_frac_of_selected"] = count / max(
                selected_count, 1.0
            )
        metrics["igsd/sampled_pair_funnel_scoreable_frac_of_retrieval_nonempty"] = (
            sampled_funnel_stats["scoreable"]
            / max(sampled_funnel_stats["both_retrieval_nonempty"], 1.0)
        )
        metrics["igsd/sampled_pair_funnel_pair_ig_valid_frac_of_scoreable"] = (
            sampled_funnel_stats["pair_ig_valid"]
            / max(sampled_funnel_stats["scoreable"], 1.0)
        )
        metrics["igsd/sampled_pair_funnel_positive_gain_frac_of_pair_ig_valid"] = (
            sampled_funnel_stats["positive_gain"]
            / max(sampled_funnel_stats["pair_ig_valid"], 1.0)
        )

        selected_plans = [
            plan
            for ex in examples
            for plan in ex.get("igsd_token_budget_plan", [])
            if bool(plan.get("selected", False))
        ]
        failure_counts: dict[str, int] = {}
        for plan in selected_plans:
            error = plan.get("pair_ig_error")
            if error:
                failure_counts[str(error)] = failure_counts.get(str(error), 0) + 1
        for error, count in failure_counts.items():
            metrics[f"igsd/sampled_pair_failure_{error}_count"] = float(count)

        association_fields = {
            "log_odds_shift": "pair_log_odds_shift",
            "prefilter_jsd": "prefilter_pair_jsd",
            "prefilter_jsd_logit_grad_abs": "prefilter_pair_jsd_student_logit_grad_abs",
            "teacher_pair_mass": "teacher_pair_mass",
            "student_pair_mass": "student_pair_mass",
            "teacher_top1_minus_sampled_log_prob": "teacher_top1_minus_sampled_log_prob",
            "student_top1_minus_sampled_log_prob": "student_top1_minus_sampled_log_prob",
            "student_top1_minus_teacher_log_prob": "student_top1_minus_teacher_log_prob",
        }

        def finite_pearson(left: list[float], right: list[float]) -> float:
            if len(left) < 2 or len(left) != len(right):
                return 0.0
            left_arr = np.asarray(left, dtype=np.float64)
            right_arr = np.asarray(right, dtype=np.float64)
            if not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
                return 0.0
            if float(left_arr.std()) == 0.0 or float(right_arr.std()) == 0.0:
                return 0.0
            return float(np.corrcoef(left_arr, right_arr)[0, 1])

        for metric_name, plan_field in association_fields.items():
            selected_values = [
                float(plan[plan_field])
                for plan in selected_plans
                if plan.get(plan_field) is not None
                and np.isfinite(float(plan[plan_field]))
            ]
            valid_pairs = [
                plan
                for plan in selected_plans
                if bool(plan.get("pair_ig_valid", False))
                and plan.get(plan_field) is not None
                and np.isfinite(float(plan[plan_field]))
                and np.isfinite(float(plan.get("pair_gain", float("nan"))))
            ]
            field_values = [float(plan[plan_field]) for plan in valid_pairs]
            gains = [float(plan["pair_gain"]) for plan in valid_pairs]
            positive = [float(bool(plan.get("pair_positive_gain", False))) for plan in valid_pairs]
            positive_values = [value for value, flag in zip(field_values, positive, strict=True) if flag > 0.0]
            nonpositive_values = [
                value for value, flag in zip(field_values, positive, strict=True) if flag <= 0.0
            ]
            prefix = f"igsd/sampled_pair_selected_{metric_name}"
            metrics[f"{prefix}_count"] = float(len(selected_values))
            metrics[f"{prefix}_mean"] = float(np.mean(selected_values)) if selected_values else 0.0
            metrics[f"{prefix}_vs_pair_gain_pearson"] = finite_pearson(field_values, gains)
            metrics[f"{prefix}_vs_positive_gain_pearson"] = finite_pearson(
                field_values, positive
            )
            metrics[f"{prefix}_positive_gain_mean"] = (
                float(np.mean(positive_values)) if positive_values else 0.0
            )
            metrics[f"{prefix}_nonpositive_gain_mean"] = (
                float(np.mean(nonpositive_values)) if nonpositive_values else 0.0
            )

        for state, values in sampled_identity_stats.items():
            prefix = f"igsd/sampled_pair_identity_{state}"
            metrics[f"{prefix}_count"] = values["count"]
            metrics[f"{prefix}_selected_count"] = values["selected"]
            metrics[f"{prefix}_record_aligned_count"] = values["record_aligned"]
            metrics[f"{prefix}_both_parsed_count"] = values["both_parsed"]
            metrics[f"{prefix}_both_retrieval_nonempty_count"] = values[
                "both_retrieval_nonempty"
            ]
            metrics[f"{prefix}_scoreable_count"] = values["scoreable"]
            metrics[f"{prefix}_exact_same_query_count"] = values["exact_same_query"]
            metrics[f"{prefix}_pair_ig_valid_count"] = values["pair_ig_valid"]
            metrics[f"{prefix}_positive_gain_count"] = values["positive_gain"]
            metrics[f"{prefix}_active_count"] = values["positive_gain"]
            metrics[f"{prefix}_pair_ig_valid_frac_of_selected"] = values[
                "pair_ig_valid"
            ] / max(values["selected"], 1.0)
            metrics[f"{prefix}_positive_gain_frac_of_pair_ig_valid"] = values[
                "positive_gain"
            ] / max(values["pair_ig_valid"], 1.0)
            metrics[f"{prefix}_gain_mean"] = values["gain_sum"] / max(
                values["pair_ig_valid"], 1.0
            )
            metrics[f"{prefix}_positive_gain_mean"] = values["positive_gain_sum"] / max(
                values["positive_gain"], 1.0
            )
            metrics[f"{prefix}_gate_sum"] = values["gate_sum"]
            metrics[f"{prefix}_gate_mean"] = values["gate_sum"] / max(
                values["positive_gain"], 1.0
            )
            for field in sampled_identity_mean_fields:
                metrics[f"{prefix}_{field}_mean"] = values.get(
                    f"{field}_sum", 0.0
                ) / max(values.get(f"{field}_count", 0.0), 1.0)
    return examples, metrics


def apply_budgeted_local_candidate_target_tilts(
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    examples: list[dict[str, Any]],
    eta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Tilt only the teacher top-1/top-2 relative mass using matched IG.

    The full teacher mass of the two candidates is invariant. All other
    teacher and student-union support entries are left untouched. A tilt is
    skipped rather than guessed when the final teacher top-k forward disagrees
    with the prefilter's detached top-1/top-2 IDs.
    """

    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "Budgeted target IDs/log-probs must have identical shapes, got "
            f"{teacher_topk_ids.shape=} and {teacher_topk_log_probs.shape=}"
        )
    if teacher_topk_ids.shape[0] != len(examples):
        raise ValueError(
            "Budgeted target examples must align with teacher top-k rows, got "
            f"{len(examples)=} and {teacher_topk_ids.shape[0]=}"
        )
    if teacher_topk_ids.ndim != 3 or teacher_topk_ids.shape[-1] < 4:
        raise ValueError(
            "Budgeted target tilting requires a teacher/student top-k union with at least two teacher entries"
        )
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError(f"Budgeted target eta must be finite and non-negative, got {eta}")

    tilted = teacher_topk_log_probs.clone()
    applied = 0
    mismatch = 0
    requested = 0
    utility_gaps: list[float] = []
    for row_idx, ex in enumerate(examples):
        for tilt in ex.get("igsd_budgeted_target_tilts", []):
            requested += 1
            position = int(tilt["action_query_position"])
            if position < 0 or position >= tilted.shape[1]:
                mismatch += 1
                continue
            expected_top1 = int(tilt["teacher_top1_id"])
            expected_top2 = int(tilt["teacher_top2_id"])
            if (
                int(teacher_topk_ids[row_idx, position, 0].item()) != expected_top1
                or int(teacher_topk_ids[row_idx, position, 1].item()) != expected_top2
            ):
                mismatch += 1
                continue
            pair_log_probs = tilted[row_idx, position, :2]
            if not bool(torch.isfinite(pair_log_probs).all()):
                mismatch += 1
                continue
            utilities = pair_log_probs.new_tensor(
                [float(tilt["top1_ig"]), float(tilt["top2_ig"])]
            )
            if not bool(torch.isfinite(utilities).all()):
                mismatch += 1
                continue
            pair_log_mass = torch.logsumexp(pair_log_probs, dim=0)
            tilted_pair = pair_log_mass + torch.log_softmax(pair_log_probs + eta * utilities, dim=0)
            tilted[row_idx, position, :2] = tilted_pair
            applied += 1
            utility_gaps.append(float(utilities[1].item() - utilities[0].item()))
    return tilted, {
        "igsd/budgeted_candidate_target_tilt_requested_count": float(requested),
        "igsd/budgeted_candidate_target_tilt_applied_count": float(applied),
        "igsd/budgeted_candidate_target_tilt_mismatch_count": float(mismatch),
        "igsd/budgeted_candidate_target_tilt_applied_frac": float(applied / max(requested, 1)),
        "igsd/budgeted_candidate_top2_minus_top1_ig_mean": float(np.mean(utility_gaps))
        if utility_gaps
        else 0.0,
    }


def build_student_bc_batch(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    lambda_coef: float,
    distill_span: str = "full_tool_call",
    distill_target: str = "teacher_action",
    teacher_log_probs: torch.Tensor | None = None,
    teacher_topk_ids: torch.Tensor | None = None,
    teacher_topk_log_probs: torch.Tensor | None = None,
    teacher_topk_mask: torch.Tensor | None = None,
    token_weight_mode: str = "uniform",
    preserve_zero_token_gate_rows: bool = False,
    distill_mode: str | None = None,
) -> DataProto | None:
    """Build unprivileged student-prefix distillation samples."""

    if not examples:
        return None
    if distill_span not in {"full_tool_call", "query_only"}:
        raise ValueError(
            f"Unsupported IGSD distill_span {distill_span!r}; supported spans are full_tool_call/query_only"
        )
    token_weight_mode = str(token_weight_mode).lower()
    sampled_action_mode = token_weight_mode == "sampled_action_pair_ig"
    if distill_mode is None:
        # Keep direct callers backward compatible. The trainer passes the
        # explicit objective so top-k JSD is not mistaken for pair JSD merely
        # because a compact test support happens to have width two.
        candidate_pair_mode = sampled_action_mode and (
            teacher_topk_ids is None or teacher_topk_ids.shape[-1] == 2
        )
    else:
        candidate_pair_mode = sampled_action_mode and str(distill_mode).lower() == "candidate_pair_jsd"
    if token_weight_mode not in {
        "uniform",
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    }:
        raise ValueError(
            "Unsupported IGSD token_weight_mode: "
            f"{token_weight_mode!r}; expected 'uniform', 'prefix_intervention_ig', or "
            "'budgeted_local_candidate_ig', or 'sampled_action_pair_ig'"
        )
    if token_weight_mode in {
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    } and distill_target != "student_on_policy":
        raise ValueError(
            "token intervention weights require distill_target='student_on_policy'"
        )
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    prompt_rows = [ex["prefix_ids"] for ex in examples]
    response_rows: list[list[int]] = []
    query_mask_rows: list[list[int]] = []
    distill_mask_rows: list[list[int]] = []
    token_gain_rows: list[list[float]] = []
    token_gate_rows: list[list[float]] = []
    token_valid_rows: list[list[bool]] = []
    token_delta_valid_rows: list[list[bool]] = []
    for ex in examples:
        response_ids, query_mask = _distill_action_ids_and_query_mask(ex, tokenizer, distill_target)
        response_rows.append(response_ids)
        query_mask_rows.append(query_mask)
        if distill_span == "query_only":
            distill_mask_rows.append(query_mask)
        else:
            distill_mask_rows.append([1] * len(response_ids))
        if token_weight_mode in {
            "prefix_intervention_ig",
            "budgeted_local_candidate_ig",
            "sampled_action_pair_ig",
        }:
            token_gains = [float(value) for value in ex.get("igsd_token_gains", [])]
            token_gates = [float(value) for value in ex.get("igsd_token_gate", [])]
            token_valid = [bool(value) for value in ex.get("igsd_token_gate_valid", [])]
            token_delta_valid = [
                bool(value) for value in ex.get("igsd_token_delta_valid", token_valid)
            ]
            expected = sum(int(value) for value in query_mask)
            if not (
                len(token_gains)
                == len(token_gates)
                == len(token_valid)
                == len(token_delta_valid)
                == expected
            ):
                raise ValueError(
                    "Token-intervention fields must align with sampled query tokens: "
                    f"{len(token_gains)=}, {len(token_gates)=}, {len(token_valid)=}, "
                    f"{len(token_delta_valid)=}, {expected=}"
                )
            if not all(np.isfinite(value) for value in token_gains):
                raise ValueError("Token-intervention gains must all be finite")
            if not all(np.isfinite(value) and value >= 0.0 for value in token_gates):
                raise ValueError("Token-intervention gates must all be finite and non-negative")
            token_gain_rows.append(token_gains)
            token_gate_rows.append(token_gates)
            token_valid_rows.append(token_valid)
            token_delta_valid_rows.append(token_delta_valid)
            has_positive_token_gate = any(
                is_valid and gate_value > 0.0
                for is_valid, gate_value in zip(token_valid, token_gates, strict=True)
            )
            has_positive_turn_weight = float(ex.get("ig_turn_weight", ex["ig_gate"])) > 0.0
            if (
                not preserve_zero_token_gate_rows
                and (not has_positive_token_gate or not has_positive_turn_weight)
            ):
                # No positive composed weight means this row cannot contribute
                # OPD, including through full-tool-call structure tokens.
                distill_mask_rows[-1] = [0] * len(response_ids)
    keep = [
        idx
        for idx, (prompt, response, distill_mask) in enumerate(
            zip(prompt_rows, response_rows, distill_mask_rows, strict=True)
        )
        if prompt and response and any(distill_mask)
    ]
    if not keep:
        return None
    # Keep argument validation after the legacy empty-row filter.  Older IGSD
    # modes historically returned ``None`` when every example was filtered;
    # moving these checks ahead of that return would turn malformed auxiliary
    # tensors into a new error for those modes.
    if (teacher_topk_ids is None) != (teacher_topk_log_probs is None):
        raise ValueError("teacher_topk_ids and teacher_topk_log_probs must be provided together")
    if sampled_action_mode and teacher_topk_ids is None:
        raise ValueError(
            "sampled_action_pair_ig requires top-k IDs/log-probabilities; "
            "pass the output of the sampled-action target builder"
        )
    if teacher_topk_mask is not None and teacher_topk_ids is None:
        raise ValueError("teacher_topk_mask requires teacher_topk_ids and teacher_topk_log_probs")
    if teacher_topk_mask is not None and teacher_topk_mask.ndim != 2:
        raise ValueError(
            "teacher_topk_mask must have shape [batch, response], got "
            f"{teacher_topk_mask.shape}"
        )
    if teacher_topk_mask is not None and teacher_topk_ids is not None:
        if teacher_topk_mask.shape[:2] != teacher_topk_ids.shape[:2]:
            raise ValueError(
                "teacher_topk_mask must align with teacher_topk_ids, got "
                f"{teacher_topk_mask.shape=} and {teacher_topk_ids.shape=}"
            )
    if candidate_pair_mode and teacher_topk_ids is not None:
        if teacher_topk_ids.ndim != 3:
            raise ValueError(
                "sampled_action_pair_ig candidate IDs must have shape [batch, response, 2], got "
                f"{teacher_topk_ids.shape}"
            )
        if teacher_topk_ids.shape[-1] != 2:
            raise ValueError(
                "sampled_action_pair_ig requires exactly two candidate IDs per position, got "
                f"{teacher_topk_ids.shape[-1]}"
            )
    examples = [examples[idx] for idx in keep]
    prompt_rows = [prompt_rows[idx] for idx in keep]
    response_rows = [response_rows[idx] for idx in keep]
    query_mask_rows = [query_mask_rows[idx] for idx in keep]
    distill_mask_rows = [distill_mask_rows[idx] for idx in keep]
    if token_weight_mode in {
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    }:
        token_gain_rows = [token_gain_rows[idx] for idx in keep]
        token_gate_rows = [token_gate_rows[idx] for idx in keep]
        token_valid_rows = [token_valid_rows[idx] for idx in keep]
        token_delta_valid_rows = [token_delta_valid_rows[idx] for idx in keep]
    if teacher_log_probs is not None:
        teacher_log_probs = teacher_log_probs[keep]
    if teacher_topk_ids is not None:
        teacher_topk_ids = teacher_topk_ids[keep]
    if teacher_topk_log_probs is not None:
        teacher_topk_log_probs = teacher_topk_log_probs[keep]
    if teacher_topk_mask is not None:
        teacher_topk_mask = teacher_topk_mask[keep]
    max_prompt = max(map(len, prompt_rows))
    max_response = max(map(len, response_rows))

    prompts = torch.full((len(examples), max_prompt), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((len(examples), max_prompt), dtype=torch.long)
    responses = torch.full((len(examples), max_response), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(examples), max_response), dtype=torch.long)
    query_token_mask = torch.zeros((len(examples), max_response), dtype=torch.bool)
    distill_span_mask = torch.zeros((len(examples), max_response), dtype=torch.long)
    token_gain_tensor = torch.zeros((len(examples), max_response), dtype=torch.float32)
    token_gate_tensor = torch.zeros((len(examples), max_response), dtype=torch.float32)
    token_valid_tensor = torch.zeros((len(examples), max_response), dtype=torch.bool)
    token_delta_valid_tensor = torch.zeros((len(examples), max_response), dtype=torch.bool)
    for idx, (prompt, response, query_mask, distill_mask) in enumerate(
        zip(prompt_rows, response_rows, query_mask_rows, distill_mask_rows, strict=True)
    ):
        prompts[idx, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
        prompt_mask[idx, -len(prompt) :] = 1
        responses[idx, : len(response)] = torch.tensor(response, dtype=torch.long)
        response_mask[idx, : len(response)] = 1
        query_token_mask[idx, : len(query_mask)] = torch.tensor(query_mask, dtype=torch.bool)
        distill_span_mask[idx, : len(distill_mask)] = torch.tensor(distill_mask, dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0) * attention_mask
    gate = torch.tensor([ex["ig_gate"] for ex in examples], dtype=torch.float32)[:, None]
    turn_weight = torch.tensor(
        [ex.get("ig_turn_weight", ex["ig_gate"]) for ex in examples], dtype=torch.float32
    )[:, None]
    if token_weight_mode in {
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    }:
        for row_idx, (query_mask, gains, gates, valid, delta_valid) in enumerate(
            zip(
                query_mask_rows,
                token_gain_rows,
                token_gate_rows,
                token_valid_rows,
                token_delta_valid_rows,
                strict=True,
            )
        ):
            query_positions = [idx for idx, value in enumerate(query_mask) if value]
            for position, gain, gate_value, is_valid, delta_is_valid in zip(
                query_positions, gains, gates, valid, delta_valid, strict=True
            ):
                token_gain_tensor[row_idx, position] = gain
                token_gate_tensor[row_idx, position] = gate_value
                token_valid_tensor[row_idx, position] = is_valid
                token_delta_valid_tensor[row_idx, position] = delta_is_valid
        query_weights = token_gate_tensor * token_valid_tensor.float() * query_token_mask.float()
        if distill_span == "full_tool_call":
            structural_weights = distill_span_mask.float() * (~query_token_mask).float()
            distill_weights = (query_weights + structural_weights) * turn_weight * float(lambda_coef)
        else:
            distill_weights = query_weights * turn_weight * float(lambda_coef)
    else:
        distill_weights = distill_span_mask.float() * turn_weight * float(lambda_coef)

    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "igsd_policy_mask": torch.zeros_like(response_mask, dtype=torch.bool),
        "igsd_query_token_mask": query_token_mask,
        "igsd_query_distill_mask": distill_span_mask.bool(),
        "igsd_distill_weights": distill_weights,
        "igsd_gate": gate.expand_as(response_mask).clone(),
        "igsd_turn_weight": turn_weight.expand_as(response_mask).clone(),
        "igsd_ig_student": torch.tensor([ex["ig_student"] for ex in examples])[:, None].expand_as(
            response_mask
        ).float(),
        "igsd_ig_teacher": torch.tensor([ex["ig_teacher"] for ex in examples])[:, None].expand_as(
            response_mask
        ).float(),
    }
    if token_weight_mode in {
        "prefix_intervention_ig",
        "budgeted_local_candidate_ig",
        "sampled_action_pair_ig",
    }:
        tensors.update(
            {
                "igsd_token_gain": token_gain_tensor,
                "igsd_token_gate": token_gate_tensor,
                "igsd_token_gate_valid": token_valid_tensor,
                "igsd_token_delta_valid": token_delta_valid_tensor,
            }
        )
    if preserve_zero_token_gate_rows and distill_target == "student_on_policy":
        tensors["igsd_opd_row_eligible"] = torch.ones(len(examples), dtype=torch.bool)
    if teacher_log_probs is not None:
        if teacher_log_probs.shape[0] != len(examples):
            raise ValueError(
                f"teacher_log_probs batch size {teacher_log_probs.shape[0]} does not match examples {len(examples)}"
            )
        if teacher_log_probs.shape[1] < max_response:
            pad = torch.zeros(
                (len(examples), max_response - teacher_log_probs.shape[1]),
                dtype=teacher_log_probs.dtype,
                device=teacher_log_probs.device,
            )
            teacher_log_probs = torch.cat([teacher_log_probs, pad], dim=1)
        tensors["igsd_teacher_log_probs"] = teacher_log_probs[:, :max_response].float()
    if teacher_topk_ids is not None and teacher_topk_log_probs is not None:
        if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
            raise ValueError(
                "teacher_topk_ids and teacher_topk_log_probs must have identical shapes, got "
                f"{teacher_topk_ids.shape} and {teacher_topk_log_probs.shape}"
            )
        if teacher_topk_ids.shape[0] != len(examples):
            raise ValueError(
                f"top-k target batch size {teacher_topk_ids.shape[0]} does not match examples {len(examples)}"
            )
        if sampled_action_mode and teacher_topk_mask is not None:
            if teacher_topk_mask.shape[1] > max_response:
                extra_active = teacher_topk_mask[:, max_response:].bool()
                if bool(extra_active.any()):
                    first = extra_active.nonzero(as_tuple=False)[0].tolist()
                    raise ValueError(
                        "sampled_action_pair_ig candidate mask contains an active position beyond "
                        f"the sampled response width: row={first[0]}, position={max_response + first[1]}"
                    )
        if sampled_action_mode and teacher_topk_mask is None:
            # A compact pair target without its active-position mask cannot be
            # distinguished from zero-filled padding.  Do not allow positive
            # token weights to reach the candidate-JSD loss in that state.
            if bool((distill_weights > 0.0).any()):
                raise ValueError(
                    "sampled_action_pair_ig requires teacher_topk_mask whenever any "
                    "token has a positive distillation weight"
                )
        if teacher_topk_ids.shape[1] < max_response:
            pad_width = max_response - teacher_topk_ids.shape[1]
            teacher_topk_ids = torch.cat(
                [
                    teacher_topk_ids,
                    torch.zeros(
                        (len(examples), pad_width, teacher_topk_ids.shape[-1]),
                        dtype=teacher_topk_ids.dtype,
                        device=teacher_topk_ids.device,
                    ),
                ],
                dim=1,
            )
            teacher_topk_log_probs = torch.cat(
                [
                    teacher_topk_log_probs,
                    torch.zeros(
                        (len(examples), pad_width, teacher_topk_log_probs.shape[-1]),
                        dtype=teacher_topk_log_probs.dtype,
                        device=teacher_topk_log_probs.device,
                    ),
                ],
                dim=1,
            )
        tensors["igsd_topk_candidate_ids"] = teacher_topk_ids[:, :max_response].long()
        tensors["igsd_topk_teacher_log_probs"] = teacher_topk_log_probs[:, :max_response].float()
        tensors["igsd_topk_positions"] = torch.arange(max_response, dtype=torch.long).unsqueeze(0).expand(
            len(examples), -1
        )
        # Top-k divergence only needs logits at positions that can contribute
        # to the configured distillation span.  Keeping this separate from
        # response_mask prevents actor updates from materializing full-vocab
        # logits for unrelated PPO trajectory tokens.
        if teacher_topk_mask is None:
            tensors["igsd_topk_mask"] = (
                distill_span_mask.bool()
                if not sampled_action_mode
                else torch.zeros_like(distill_span_mask, dtype=torch.bool)
            )
            if sampled_action_mode:
                tensors["igsd_distill_weights"] = torch.zeros_like(distill_weights)
        else:
            if teacher_topk_mask.shape[0] != len(examples):
                raise ValueError(
                    "teacher_topk_mask batch size does not match examples after filtering, got "
                    f"{teacher_topk_mask.shape[0]} and {len(examples)}"
                )
            if teacher_topk_mask.shape[1] < max_response:
                teacher_topk_mask = torch.cat(
                    [
                        teacher_topk_mask,
                        torch.zeros(
                            (len(examples), max_response - teacher_topk_mask.shape[1]),
                            dtype=torch.bool,
                            device=teacher_topk_mask.device,
                        ),
                    ],
                    dim=1,
                )
            pair_mask = teacher_topk_mask[:, :max_response].bool()
            if sampled_action_mode:
                positive_weights = distill_weights > 0.0
                # A positive token weight with no active pair target indicates
                # a planner/packing mismatch.  Raising here is preferable to
                # silently deleting the only supervision signal for that token.
                missing_targets = positive_weights & ~pair_mask
                if bool(missing_targets.any()):
                    first = missing_targets.nonzero(as_tuple=False)[0].tolist()
                    raise ValueError(
                        "sampled_action_pair_ig has positive distillation weight without an "
                        "active candidate pair at "
                        f"row={first[0]}, position={first[1]}"
                    )

                active_positions = pair_mask & response_mask.bool() & query_token_mask
                if bool((pair_mask & ~active_positions).any()):
                    first = (pair_mask & ~active_positions).nonzero(as_tuple=False)[0].tolist()
                    raise ValueError(
                        "sampled_action_pair_ig candidate mask must be restricted to sampled "
                        f"query response positions, got row={first[0]}, position={first[1]}"
                    )

                if candidate_pair_mode:
                    active_ids = teacher_topk_ids[:, :max_response][pair_mask]
                    active_log_probs = teacher_topk_log_probs[:, :max_response][pair_mask]
                    if active_ids.numel():
                        if bool((active_ids < 0).any()):
                            raise ValueError(
                                "sampled_action_pair_ig active candidate IDs must be non-negative"
                            )
                        if bool(active_ids[:, 0].eq(active_ids[:, 1]).any()):
                            raise ValueError(
                                "sampled_action_pair_ig active candidate pairs must be distinct"
                            )
                        if not bool(torch.isfinite(active_log_probs).all()):
                            raise ValueError(
                                "sampled_action_pair_ig active candidate log-probabilities must be finite"
                            )
                        if bool(active_log_probs.gt(0.0).any()):
                            raise ValueError(
                                "sampled_action_pair_ig active candidate log-probabilities must be <= 0"
                            )

                # Keep the compact-logit mask and the positive loss mask
                # exactly identical.  The intersection handles a zero global
                # scale (for example a zero lambda) without activating a pair,
                # while the check above catches the dangerous opposite case.
                pair_mask = pair_mask & positive_weights
                distill_weights = distill_weights * pair_mask.to(distill_weights.dtype)
                tensors["igsd_distill_weights"] = distill_weights
                if not torch.equal(pair_mask, tensors["igsd_distill_weights"] > 0.0):
                    raise AssertionError(
                        "sampled_action_pair_ig candidate mask and positive distillation weights diverged"
                    )
            tensors["igsd_topk_mask"] = pair_mask

    td = TensorDict(tensors, batch_size=[len(examples)])
    non_tensor = {
        "igsd_source_sample_idx": np.array([ex["source_sample_idx"] for ex in examples], dtype=object),
        "igsd_source_turn_idx": np.array([ex["source_turn_idx"] for ex in examples], dtype=object),
        "igsd_source_turn_count": np.array([ex.get("source_turn_count", 1) for ex in examples], dtype=object),
        "igsd_student_query": np.array([ex["student_query"] for ex in examples], dtype=object),
        "igsd_teacher_query": np.array([ex["teacher_query"] for ex in examples], dtype=object),
    }
    return DataProto(batch=td, non_tensor_batch=non_tensor)


def build_action_logprob_batch(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    prompt_key: str,
    distill_target: str = "teacher_action",
) -> DataProto | None:
    """Build prompt -> shared distillation action samples for logprob scoring.

    ``prompt_key`` selects the context whose probability should be evaluated:
    - ``teacher_prompt_ids``: privileged hindsight teacher context.
    - ``prefix_ids``: unprivileged student context.
    """

    if not examples:
        return None
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    prompt_rows = [list(ex.get(prompt_key, [])) for ex in examples]
    if distill_target == "teacher_action":
        response_rows = [
            tokenizer.encode(
                canonical_search_action_queries(
                    ex.get("teacher_queries", [ex.get("teacher_query", "")])
                ),
                add_special_tokens=False,
            )
            for ex in examples
        ]
        query_mask_rows: list[list[int]] | None = None
    else:
        action_rows_and_masks = [
            _distill_action_ids_and_query_mask(ex, tokenizer, distill_target) for ex in examples
        ]
        response_rows = [action_ids for action_ids, _ in action_rows_and_masks]
        query_mask_rows = [query_mask for _, query_mask in action_rows_and_masks]
    keep = [
        idx
        for idx, (prompt, response) in enumerate(zip(prompt_rows, response_rows, strict=True))
        if prompt and response
    ]
    if not keep:
        return None
    prompt_rows = [prompt_rows[idx] for idx in keep]
    response_rows = [response_rows[idx] for idx in keep]
    if query_mask_rows is not None:
        query_mask_rows = [query_mask_rows[idx] for idx in keep]
    max_prompt = max(map(len, prompt_rows))
    max_response = max(map(len, response_rows))

    prompts = torch.full((len(keep), max_prompt), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((len(keep), max_prompt), dtype=torch.long)
    responses = torch.full((len(keep), max_response), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(keep), max_response), dtype=torch.long)
    action_query_mask = torch.zeros((len(keep), max_response), dtype=torch.bool)
    for row_idx, (prompt, response) in enumerate(zip(prompt_rows, response_rows, strict=True)):
        prompts[row_idx, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
        prompt_mask[row_idx, -len(prompt) :] = 1
        responses[row_idx, : len(response)] = torch.tensor(response, dtype=torch.long)
        response_mask[row_idx, : len(response)] = 1
        if query_mask_rows is not None:
            query_mask = query_mask_rows[row_idx]
            if len(query_mask) != len(response):
                raise ValueError(
                    "Student action query mask must align with the on-policy response, "
                    f"got {len(query_mask)=} and {len(response)=}"
                )
            action_query_mask[row_idx, : len(query_mask)] = torch.tensor(query_mask, dtype=torch.bool)

    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0) * attention_mask
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
    }
    if query_mask_rows is not None:
        tensors["igsd_action_query_mask"] = action_query_mask
    td = TensorDict(tensors, batch_size=[len(keep)])
    return DataProto(batch=td, non_tensor_batch={"igsd_kept_example_idx": np.array(keep, dtype=object)})
