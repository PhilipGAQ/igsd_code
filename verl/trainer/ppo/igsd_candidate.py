# Copyright 2026 IGSD Contributors
# Licensed under the Apache License, Version 2.0.
"""Small, dependency-free rule for allocating candidate verification."""

from __future__ import annotations

import math


def is_eligible_disagreement(
    teacher_token_id: int,
    sampled_token_id: int,
    pair_log_odds_shift: float,
    *,
    epsilon: float = 1e-6,
) -> bool:
    """Admit either shift direction, then let executed evidence select pairs.

    ``epsilon`` only filters numerically indistinguishable probability shifts.
    The caller also checks all candidate log probabilities for finiteness.
    """

    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    return (
        teacher_token_id != sampled_token_id
        and math.isfinite(pair_log_odds_shift)
        and abs(pair_log_odds_shift) > epsilon
    )
