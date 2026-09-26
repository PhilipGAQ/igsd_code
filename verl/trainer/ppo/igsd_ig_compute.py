# Copyright 2026 The SearchAgent-Zero authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Information Gain computation for IGSD.

This module implements the counterfactual IG computation:
  IG_t = mean_k log π(a*_k | C_real,t) - (1/N) Σ_j mean_k log π(a*_k | C_rand,j,t)

where:
  - a* is the ground-truth answer (or final correct answer from a success sibling)
  - C_real,t is the context with real retrieved documents at turn t
  - C_rand,j,t is the context with random documents replacing real ones

The IG is a model-based contrast for answer support from retrieved versus
random documents; it is not a causal estimator of terminal answer quality.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SearchTurn:
    """Represents a single search turn within a multi-turn trajectory."""

    turn_index: int
    # Token-level positions in the response tensor. ``query_start/query_end``
    # are kept for compatibility and cover the complete assistant search
    # action (thought + tool call), not only the serialized query value.
    query_start: int  # start of assistant search action (inclusive)
    query_end: int  # end of assistant search action (exclusive)
    tool_response_start: int  # start of tool response tokens (inclusive)
    tool_response_end: int  # end of tool response tokens (exclusive)
    # Exact JSON query value span when it can be aligned back to response IDs.
    query_value_start: int = -1
    query_value_end: int = -1
    # Per-query values and token spans.  The legacy scalar fields above remain
    # populated as a conservative bounding span for old single-query callers.
    query_texts: list[str] = field(default_factory=list)
    query_value_spans: list[tuple[int, int]] = field(default_factory=list)
    # True only for the ASearcher schema: a direct search-call dictionary with
    # a non-empty ``query_list`` of non-empty strings. Legacy callers may still
    # consume normalized scalar/malformed forms when strict mode is disabled.
    query_schema_valid: bool = True
    # Extracted text
    query_text: str = ""
    tool_response_text: str = ""
    # Parsed documents from tool response
    documents: list[str] = field(default_factory=list)
    # Per-query document groups when the tool response contains the ASearcher
    # result separator.  Legacy callers can continue using ``documents``.
    documents_by_query: list[list[str]] = field(default_factory=list)


@dataclass
class ParsedTrajectory:
    """A parsed multi-turn trajectory with identified search turns."""

    sample_index: int
    search_turns: list[SearchTurn] = field(default_factory=list)
    answer_text: str = ""
    # Token positions for the final answer span
    answer_start: int = -1
    answer_end: int = -1


@dataclass
class IGResult:
    """Per-sample IG computation result."""

    sample_index: int
    # Per-turn IG values
    ig_values: list[float] = field(default_factory=list)
    # Token-level IG tensor (response_len,) with IG value at query end positions
    ig_per_token: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

# Hermes tool call format: <tool_call>{"name": "search", "arguments": {...}}</tool_call>
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
JSON_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)


def _query_value_char_spans(tool_call_match: re.Match[str]) -> list[tuple[int, int]]:
    """Locate query-list string contents inside a matched tool call.

    The returned offsets are relative to the decoded assistant span and omit
    the surrounding JSON quotes.  A separate span is returned for every
    literal in ``query_list`` so commas, brackets, and JSON scaffolding are not
    accidentally treated as query tokens.
    """

    payload = tool_call_match.group(1)
    key_match = re.search(r'"query_list"\s*:\s*', payload)
    if key_match is None:
        return []
    value_text = payload[key_match.end() :]
    stripped = value_text.lstrip()
    leading_space = len(value_text) - len(stripped)
    if not stripped:
        return []

    try:
        parsed_value, consumed = json.JSONDecoder().raw_decode(stripped)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed_value, str) and not (
        isinstance(parsed_value, list) and parsed_value and all(isinstance(item, str) for item in parsed_value)
    ):
        return []
    raw_value = stripped[:consumed]
    literals = list(JSON_STRING_PATTERN.finditer(raw_value))
    if not literals:
        return []

    payload_offset = key_match.end() + leading_space
    spans: list[tuple[int, int]] = []
    for literal in literals:
        start = tool_call_match.start(1) + payload_offset + literal.start() + 1
        end = tool_call_match.start(1) + payload_offset + literal.end() - 1
        if end > start:
            spans.append((start, end))
    return spans


def _query_value_char_span(tool_call_match: re.Match[str]) -> tuple[int, int] | None:
    """Legacy bounding-span wrapper for single-query callers."""

    spans = _query_value_char_spans(tool_call_match)
    if not spans:
        return None
    return spans[0][0], spans[-1][1]


def _normalise_query_list(value: Any) -> list[str]:
    """Convert a search ``query_list`` to clean query strings.

    ASearcher's tool schema is list-valued, while older Search-R1 examples may
    emit a scalar.  Empty/malformed entries are dropped rather than creating
    phantom query slots; the caller can still use the legacy flattened string.
    """

    if isinstance(value, list):
        items = value
    else:
        items = [value]
    queries: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("query", item.get("text", item.get("q", "")))).strip()
        else:
            text = str(item).strip() if item is not None else ""
        if text:
            queries.append(text)
    return queries


def _char_span_to_token_span(
    text: str,
    token_ids: list[int],
    char_start: int,
    char_end: int,
    tokenizer: Any,
) -> tuple[int, int] | None:
    """Map a decoded character span back to its token interval."""

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = encoded.get("input_ids")
        offsets = encoded.get("offset_mapping")
        if encoded_ids is not None and offsets is not None and list(encoded_ids) == list(token_ids):
            indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if int(end) > char_start and int(start) < char_end
            ]
            if indices:
                return indices[0], indices[-1] + 1
    except (NotImplementedError, TypeError, ValueError, AttributeError):
        pass

    # Conservative fallback for slow tokenizers. Re-encoding decoded prefixes
    # can shift a boundary by one merged token, but it is still preferable to
    # silently treating the whole thought/tool-call span as the query.
    try:
        prefix_ids = tokenizer.encode(text[:char_start], add_special_tokens=False)
        prefix_plus_query_ids = tokenizer.encode(text[:char_end], add_special_tokens=False)
    except (TypeError, ValueError, AttributeError):
        return None
    start = min(len(prefix_ids), len(token_ids))
    end = min(max(len(prefix_plus_query_ids), start), len(token_ids))
    return (start, end) if end > start else None


def parse_response_text(response_text: str) -> tuple[list[dict], str]:
    """Parse a multi-turn response into tool calls and final answer.

    Returns:
        (tool_calls, answer_text) where tool_calls is a list of dicts with
        'query_text' and 'raw_match' keys.
    """
    tool_calls = []
    for match in TOOL_CALL_PATTERN.finditer(response_text):
        try:
            call_data = json.loads(match.group(1))
            if isinstance(call_data, list):
                call_data = call_data[0] if call_data and isinstance(call_data[0], dict) else None
            if not isinstance(call_data, dict):
                continue
            name = call_data.get("name", "")
            arguments = call_data.get("arguments", {})
            if name == "search":
                if isinstance(arguments, dict):
                    query_texts = _normalise_query_list(arguments.get("query_list", []))
                    query_text = " ".join(query_texts)
                else:
                    query_texts = []
                    query_text = str(arguments)
            else:
                query_texts = []
                query_text = json.dumps(arguments)
            tool_calls.append({
                "name": name,
                "query_text": query_text,
                "query_texts": query_texts,
                "match_start": match.start(),
                "match_end": match.end(),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    answer_text = ""
    answer_matches = list(ANSWER_PATTERN.finditer(response_text))
    if answer_matches:
        answer_text = answer_matches[-1].group(1).strip()

    return tool_calls, answer_text


def parse_trajectory_tokens(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    sample_index: int = 0,
    result_separator: str = "\n-*-*-\n",
) -> ParsedTrajectory:
    """Parse a single sample's response tokens into a structured trajectory.

    This identifies search turns by finding assistant-generated spans (response_mask=1)
    that contain tool calls, and the subsequent tool-response spans (response_mask=0).

    Args:
        response_ids: (response_len,) token ids
        response_mask: (response_len,) binary mask (1=assistant, 0=tool/padding)
        tokenizer: tokenizer for decoding
        sample_index: index in the batch

    Returns:
        ParsedTrajectory with identified search turns
    """
    # Decode full response text
    valid_ids = response_ids[response_mask.bool()] if response_mask.any() else response_ids
    full_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=False)

    # Identify spans: contiguous regions of mask=1 (assistant) and mask=0 (tool response)
    mask_np = response_mask.cpu().numpy()
    spans = []  # list of (start, end, is_assistant)
    if len(mask_np) == 0:
        return ParsedTrajectory(sample_index=sample_index)

    current_val = mask_np[0]
    span_start = 0
    for i in range(1, len(mask_np)):
        if mask_np[i] != current_val:
            spans.append((span_start, i, bool(current_val)))
            span_start = i
            current_val = mask_np[i]
    spans.append((span_start, len(mask_np), bool(current_val)))

    # Match assistant spans containing tool_calls with subsequent tool_response spans
    search_turns = []
    turn_idx = 0
    for span_i, (start, end, is_assistant) in enumerate(spans):
        if not is_assistant:
            continue
        # Decode this assistant span
        span_ids = response_ids[start:end].tolist()
        span_text = tokenizer.decode(span_ids, skip_special_tokens=False)

        # Check if this span contains a tool call
        tool_call_match = TOOL_CALL_PATTERN.search(span_text)
        if tool_call_match is None:
            continue

        query_value_spans: list[tuple[int, int]] = []
        query_char_spans = _query_value_char_spans(tool_call_match)
        for char_start, char_end in query_char_spans:
            local_query_span = _char_span_to_token_span(
                text=span_text,
                token_ids=span_ids,
                char_start=char_start,
                char_end=char_end,
                tokenizer=tokenizer,
            )
            if local_query_span is not None:
                query_value_spans.append(
                    (start + local_query_span[0], start + local_query_span[1])
                )
        if query_value_spans:
            query_value_start = min(span[0] for span in query_value_spans)
            query_value_end = max(span[1] for span in query_value_spans)
        else:
            query_value_start = -1
            query_value_end = -1

        # Parse the tool call
        try:
            call_data = json.loads(tool_call_match.group(1))
            # Model may generate a list wrapper, e.g. [{"name": "search", ...}]
            had_list_wrapper = isinstance(call_data, list)
            if had_list_wrapper:
                if len(call_data) > 0 and isinstance(call_data[0], dict):
                    call_data = call_data[0]
                else:
                    continue
            if not isinstance(call_data, dict):
                continue
            name = call_data.get("name", "")
            if name != "search":
                continue
            arguments = call_data.get("arguments", {})
            if not isinstance(arguments, dict):
                # Model generated malformed arguments (e.g., a list instead of dict)
                query_texts = []
                query_text = str(arguments)
                query_schema_valid = False
            else:
                raw_query_list = arguments.get("query_list", [])
                query_schema_valid = bool(
                    not had_list_wrapper
                    and isinstance(raw_query_list, list)
                    and raw_query_list
                    and all(isinstance(item, str) and item.strip() for item in raw_query_list)
                )
                query_texts = _normalise_query_list(raw_query_list)
                query_text = " ".join(query_texts)
        except (json.JSONDecodeError, KeyError):
            continue

        # Find the tool response span (next span with is_assistant=False)
        tool_resp_start = end
        tool_resp_end = end
        if span_i + 1 < len(spans):
            next_start, next_end, next_is_asst = spans[span_i + 1]
            if not next_is_asst:
                tool_resp_start = next_start
                tool_resp_end = next_end

        # Decode tool response
        tool_resp_text = ""
        tool_resp_docs = []
        tool_resp_docs_by_query: list[list[str]] = []
        if tool_resp_end > tool_resp_start:
            tool_resp_ids = response_ids[tool_resp_start:tool_resp_end].tolist()
            tool_resp_text = tokenizer.decode(tool_resp_ids, skip_special_tokens=False)
            tool_resp_docs = _extract_documents_from_tool_response(tool_resp_text)
            tool_resp_docs_by_query = _extract_documents_from_tool_response_by_query(
                tool_resp_text,
                len(query_texts),
                result_separator=result_separator,
            ) if query_texts else []

        search_turns.append(SearchTurn(
            turn_index=turn_idx,
            query_start=start,
            query_end=end,
            tool_response_start=tool_resp_start,
            tool_response_end=tool_resp_end,
            query_value_start=query_value_start,
            query_value_end=query_value_end,
            query_texts=query_texts,
            query_value_spans=query_value_spans,
            query_schema_valid=query_schema_valid,
            query_text=query_text,
            tool_response_text=tool_resp_text,
            documents=tool_resp_docs,
            documents_by_query=tool_resp_docs_by_query,
        ))
        turn_idx += 1

    # Find answer span (last assistant span without tool call, or containing <answer>)
    answer_text = ""
    answer_start = -1
    answer_end = -1
    for start, end, is_assistant in reversed(spans):
        if not is_assistant:
            continue
        span_ids = response_ids[start:end].tolist()
        span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
        ans_match = ANSWER_PATTERN.search(span_text)
        if ans_match:
            answer_text = ans_match.group(1).strip()
            answer_start = start
            answer_end = end
            break

    return ParsedTrajectory(
        sample_index=sample_index,
        search_turns=search_turns,
        answer_text=answer_text,
        answer_start=answer_start,
        answer_end=answer_end,
    )


def _extract_documents_from_tool_response(tool_resp_text: str) -> list[str]:
    """Extract individual documents from the tool response text.

    The tool response typically looks like:
    {"result": "Doc 1 (Title: ...) ...\nDoc 2 (Title: ...) ..."}
    """
    # The decoded span normally includes chat-template role wrappers around the
    # JSON payload. Try both the full text and the outermost JSON object.
    json_candidates = [tool_resp_text]
    json_start = tool_resp_text.find("{")
    json_end = tool_resp_text.rfind("}")
    if json_start >= 0 and json_end >= json_start:
        json_candidates.append(tool_resp_text[json_start : json_end + 1])
    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            result_text = data.get("result", "")
            if isinstance(result_text, str):
                normalized_result = " ".join(result_text.lower().split())
                if (
                    normalized_result == "no search results found."
                    or normalized_result.startswith("error:")
                    or normalized_result.startswith("search execution exception:")
                    or normalized_result.startswith("search failed after ")
                    or normalized_result.startswith("unexpected execution error:")
                    or normalized_result.startswith("result parsing failed:")
                ):
                    return []
                docs = re.split(r"(?=Doc\s+\d+\s*\(Title:)", result_text)
                return [doc.strip() for doc in docs if doc.strip()]
            if isinstance(result_text, list):
                docs = []
                for item in result_text:
                    if isinstance(item, dict):
                        docs.append(item.get("document", str(item)))
                    else:
                        docs.append(str(item))
                return docs
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    # Fallback: split by Doc N pattern in raw text
    doc_matches = list(re.finditer(r"Doc\s+\d+\s*\(Title:", tool_resp_text))
    if doc_matches:
        return [
            tool_resp_text[match.start() : doc_matches[idx + 1].start()].strip()
            if idx + 1 < len(doc_matches)
            else tool_resp_text[match.start() :].strip()
            for idx, match in enumerate(doc_matches)
        ]
    return []


def _extract_documents_from_tool_response_by_query(
    tool_resp_text: str,
    query_count: int,
    result_separator: str = "\n-*-*-\n",
) -> list[list[str]]:
    """Extract raw result groups while preserving multi-query boundaries."""

    count = max(int(query_count), 1)
    json_candidates = [tool_resp_text]
    json_start = tool_resp_text.find("{")
    json_end = tool_resp_text.rfind("}")
    if json_start >= 0 and json_end >= json_start:
        json_candidates.append(tool_resp_text[json_start : json_end + 1])
    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            result_text = data.get("result", "")
            if not isinstance(result_text, str):
                flat = _extract_documents_from_tool_response(candidate)
                return [flat] + [[] for _ in range(count - 1)]
            blocks = result_text.split(result_separator) if result_separator else [result_text]
            if len(blocks) != count:
                flat = _extract_documents_from_tool_response(candidate)
                return [flat] + [[] for _ in range(count - 1)]
            groups: list[list[str]] = []
            for block in blocks:
                # Parsing a block through the existing helper handles Doc-N
                # boundaries without treating the separator as document text.
                groups.append(_extract_documents_from_tool_response(json.dumps({"result": block})))
            return groups
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    flat = _extract_documents_from_tool_response(tool_resp_text)
    return [flat] + [[] for _ in range(count - 1)]


# ---------------------------------------------------------------------------
# Query Span Mask Generation
# ---------------------------------------------------------------------------


def build_query_span_mask(
    batch_response_ids: torch.Tensor,
    batch_response_mask: torch.Tensor,
    tokenizer: Any,
) -> tuple[torch.Tensor, list[list[SearchTurn]]]:
    """Build a token-level mask that marks query spans (tool_call content) in the response.

    Args:
        batch_response_ids: (batch_size, response_len) token ids
        batch_response_mask: (batch_size, response_len) binary mask
        tokenizer: tokenizer for decoding

    Returns:
        query_mask: (batch_size, response_len) bool tensor, True at query token positions
        all_turns: list of list of SearchTurn per sample
    """
    batch_size, response_len = batch_response_ids.shape
    device = batch_response_ids.device
    query_mask = torch.zeros(batch_size, response_len, dtype=torch.bool, device=device)
    all_turns: list[list[SearchTurn]] = []

    for i in range(batch_size):
        traj = parse_trajectory_tokens(
            response_ids=batch_response_ids[i],
            response_mask=batch_response_mask[i],
            tokenizer=tokenizer,
            sample_index=i,
        )
        all_turns.append(traj.search_turns)
        for turn in traj.search_turns:
            # Mark the query span (the assistant-generated tool_call span)
            query_mask[i, turn.query_start:turn.query_end] = True

    return query_mask, all_turns


# ---------------------------------------------------------------------------
# IG Computation
# ---------------------------------------------------------------------------


def compute_ig_for_batch(
    model: Any,
    tokenizer: Any,
    batch_response_ids: torch.Tensor,
    batch_response_mask: torch.Tensor,
    answer_texts: list[str],
    all_search_turns: list[list[SearchTurn]],
    random_doc_pool: list[str],
    num_counterfactual: int = 3,
    max_answer_tokens: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute per-turn Information Gain for a batch.

    For each sample's each search turn:
      IG_t = mean_k log π(a*_k | C_real,t) - (1/N) Σ_j mean_k log π(a*_k | C_rand,j,t)

    This is a simplified implementation that:
    1. Constructs context with real docs → compute answer log-prob
    2. Constructs context with random docs → compute answer log-prob
    3. IG = real_logprob - mean(random_logprobs)

    Args:
        model: The policy model (for forward pass)
        tokenizer: Tokenizer
        batch_response_ids: (batch_size, response_len)
        batch_response_mask: (batch_size, response_len)
        answer_texts: list of ground-truth/success answers per sample
        all_search_turns: pre-parsed search turns per sample
        random_doc_pool: pool of documents to sample from for counterfactual
        num_counterfactual: number of random doc replacements (N)
        max_answer_tokens: max tokens to consider from the answer
        device: device for computation

    Returns:
        ig_student: (batch_size, response_len) tensor with IG values at turn-end positions
        query_mask: (batch_size, response_len) bool tensor marking query spans
        metrics: dict of diagnostic metrics
    """
    batch_size, response_len = batch_response_ids.shape
    if device is None:
        device = batch_response_ids.device

    ig_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32, device=device)
    query_mask = torch.zeros(batch_size, response_len, dtype=torch.bool, device=device)

    total_turns = 0
    total_ig = 0.0
    ig_values_all = []

    for sample_idx in range(batch_size):
        turns = all_search_turns[sample_idx]
        answer_text = answer_texts[sample_idx] if sample_idx < len(answer_texts) else ""

        if not turns or not answer_text:
            continue

        # Tokenize the answer
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)[:max_answer_tokens]
        if not answer_ids:
            continue

        for turn in turns:
            # Mark query span
            query_mask[sample_idx, turn.query_start:turn.query_end] = True

            if not turn.documents:
                continue

            # Compute IG for this turn
            ig_value = _compute_single_turn_ig(
                model=model,
                tokenizer=tokenizer,
                response_ids=batch_response_ids[sample_idx],
                response_mask=batch_response_mask[sample_idx],
                turn=turn,
                answer_ids=answer_ids,
                random_doc_pool=random_doc_pool,
                num_counterfactual=num_counterfactual,
                device=device,
            )

            # Place IG value at the end of the query span (turn boundary)
            # This aligns with the turn_end_mask used in igsd_utils
            end_pos = min(turn.query_end - 1, response_len - 1)
            if end_pos >= 0:
                ig_tensor[sample_idx, end_pos] = ig_value

            total_turns += 1
            total_ig += ig_value
            ig_values_all.append(ig_value)

    metrics = {
        "ig_compute/total_turns": float(total_turns),
        "ig_compute/mean_ig": float(total_ig / max(total_turns, 1)),
        "ig_compute/positive_ig_frac": float(
            sum(1 for v in ig_values_all if v > 0) / max(len(ig_values_all), 1)
        ),
    }
    if ig_values_all:
        metrics["ig_compute/ig_std"] = float(np.std(ig_values_all))
        metrics["ig_compute/ig_max"] = float(max(ig_values_all))
        metrics["ig_compute/ig_min"] = float(min(ig_values_all))

    return ig_tensor, query_mask, metrics


@torch.no_grad()
def _compute_single_turn_ig(
    model: Any,
    tokenizer: Any,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    turn: SearchTurn,
    answer_ids: list[int],
    random_doc_pool: list[str],
    num_counterfactual: int,
    device: torch.device,
) -> float:
    """Compute IG for a single search turn.

    IG_t = logprob(answer | context_with_real_docs) - mean(logprob(answer | context_with_random_docs))

    The 'context' is: prompt + response up to tool_response_end (i.e., after seeing the docs)
    """
    if not random_doc_pool:
        return 0.0

    # Build context prefix: everything up to and including the tool response
    # This is the state after the model has "seen" the search results
    context_end = turn.tool_response_end
    context_ids = response_ids[:context_end].tolist()

    # Compute log-prob of answer given real context
    real_logprob = _answer_logprob(model, tokenizer, context_ids, answer_ids, device)

    # Compute log-probs of answer given counterfactual contexts
    counterfactual_logprobs = []
    for _ in range(num_counterfactual):
        # Sample random documents
        n_docs = max(len(turn.documents), 1)
        random_docs = _sample_random_docs(random_doc_pool, n_docs)
        # Build counterfactual tool response
        counterfactual_resp = _format_counterfactual_tool_response(random_docs)
        counterfactual_resp_ids = tokenizer.encode(counterfactual_resp, add_special_tokens=False)

        # Replace the tool response tokens in context
        context_before_tool_resp = response_ids[:turn.tool_response_start].tolist()
        cf_context_ids = context_before_tool_resp + counterfactual_resp_ids

        cf_logprob = _answer_logprob(model, tokenizer, cf_context_ids, answer_ids, device)
        counterfactual_logprobs.append(cf_logprob)

    if not counterfactual_logprobs:
        return 0.0

    mean_cf_logprob = sum(counterfactual_logprobs) / len(counterfactual_logprobs)
    ig = real_logprob - mean_cf_logprob
    return ig


@torch.no_grad()
def _answer_logprob(
    model: Any,
    tokenizer: Any,
    context_ids: list[int],
    answer_ids: list[int],
    device: torch.device,
) -> float:
    """Compute mean log-probability of answer tokens given context.

    Returns: mean log P(answer_k | context, answer_{<k})
    """
    # Concatenate context + answer
    input_ids = context_ids + answer_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Forward pass
    try:
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    except Exception as e:
        logger.warning(f"Forward pass failed in IG computation: {e}")
        return 0.0

    # Extract log-probs for answer tokens
    # The log-prob of token at position i is computed from logits at position i-1
    answer_start_pos = len(context_ids)
    answer_end_pos = len(input_ids)

    # Get logits for positions that predict answer tokens
    predict_logits = logits[0, answer_start_pos - 1:answer_end_pos - 1, :]  # (answer_len, vocab)
    answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device)

    log_probs = torch.nn.functional.log_softmax(predict_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_tensor.unsqueeze(1)).squeeze(1)  # (answer_len,)

    mean_log_prob = token_log_probs.mean().item()
    return mean_log_prob


def _sample_random_docs(doc_pool: list[str], n: int) -> list[str]:
    """Sample n random documents from the pool."""
    if len(doc_pool) <= n:
        return doc_pool[:]
    indices = np.random.choice(len(doc_pool), size=n, replace=False)
    return [doc_pool[i] for i in indices]


def _format_counterfactual_tool_response(docs: list[str]) -> str:
    """Format random documents into the same format as the tool response.

    Mimics the format produced by the search tool.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"Doc {i} (Title: Random Document {i})\n{doc}")
    result_text = "\n\n".join(parts)
    return json.dumps({"result": result_text})


# ---------------------------------------------------------------------------
# Random Document Pool Builder
# ---------------------------------------------------------------------------


def build_random_doc_pool_from_batch(
    all_search_turns: list[list[SearchTurn]],
    max_pool_size: int = 500,
) -> list[str]:
    """Build a pool of random documents from the batch's search turns.

    Collects all retrieved documents across all samples and turns in the batch.
    This provides a diverse pool for counterfactual replacement.
    """
    pool: list[str] = []
    for turns in all_search_turns:
        for turn in turns:
            pool.extend(turn.documents)

    # Deduplicate and limit size
    seen = set()
    unique_pool = []
    for doc in pool:
        doc_key = doc[:200]  # Use prefix as dedup key
        if doc_key not in seen:
            seen.add(doc_key)
            unique_pool.append(doc)

    if len(unique_pool) > max_pool_size:
        indices = np.random.choice(len(unique_pool), size=max_pool_size, replace=False)
        unique_pool = [unique_pool[i] for i in indices]

    return unique_pool


# ---------------------------------------------------------------------------
# Batch-level IG Computation (Lightweight Version for Training)
# ---------------------------------------------------------------------------


def compute_ig_lightweight(
    batch_response_ids: torch.Tensor,
    batch_response_mask: torch.Tensor,
    tokenizer: Any,
    answer_texts: list[str],
    model: Any,
    num_counterfactual: int = 3,
    max_answer_tokens: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[list[SearchTurn]], dict[str, float]]:
    """End-to-end IG computation: parse trajectories, build doc pool, compute IG.

    This is the main entry point for computing IG during training.

    Args:
        batch_response_ids: (batch_size, response_len)
        batch_response_mask: (batch_size, response_len)
        tokenizer: tokenizer
        answer_texts: ground-truth or success-sibling answers per sample
        model: policy model for forward pass
        num_counterfactual: N for counterfactual doc replacement
        max_answer_tokens: max answer tokens to consider
        device: compute device

    Returns:
        ig_tensor: (batch_size, response_len) with IG at turn-end positions
        query_mask: (batch_size, response_len) bool mask for query spans
        all_search_turns: parsed search turns per sample
        metrics: diagnostic metrics dict
    """
    batch_size = batch_response_ids.shape[0]

    # Step 1: Parse trajectories
    all_search_turns: list[list[SearchTurn]] = []
    for i in range(batch_size):
        traj = parse_trajectory_tokens(
            response_ids=batch_response_ids[i],
            response_mask=batch_response_mask[i],
            tokenizer=tokenizer,
            sample_index=i,
        )
        all_search_turns.append(traj.search_turns)

    # Step 2: Build random document pool from batch
    random_doc_pool = build_random_doc_pool_from_batch(all_search_turns)

    # Step 3: Compute IG
    ig_tensor, query_mask, metrics = compute_ig_for_batch(
        model=model,
        tokenizer=tokenizer,
        batch_response_ids=batch_response_ids,
        batch_response_mask=batch_response_mask,
        answer_texts=answer_texts,
        all_search_turns=all_search_turns,
        random_doc_pool=random_doc_pool,
        num_counterfactual=num_counterfactual,
        max_answer_tokens=max_answer_tokens,
        device=device,
    )

    # Additional metrics
    total_turns = sum(len(turns) for turns in all_search_turns)
    samples_with_turns = sum(1 for turns in all_search_turns if turns)
    metrics["ig_compute/samples_with_search_turns"] = float(samples_with_turns)
    metrics["ig_compute/total_search_turns"] = float(total_turns)
    metrics["ig_compute/mean_turns_per_sample"] = float(total_turns / max(batch_size, 1))
    metrics["ig_compute/doc_pool_size"] = float(len(random_doc_pool))

    return ig_tensor, query_mask, all_search_turns, metrics
