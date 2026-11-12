"""Shifted warm starts for receding-horizon problems.

A segment ``(start, size, count)`` describes ``count`` consecutive stages of
``size`` entries starting at ``start``. Shifting moves stage k+1 to stage k and
either holds or zeros the last stage. Entries outside every segment are kept.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch

__all__ = ["Segment", "shift_stages"]

Segment = Tuple[int, int, int]


def shift_stages(v: torch.Tensor, segments: Iterable[Segment], tail: str = "hold") -> torch.Tensor:
    """Return a copy of ``v`` with every segment shifted one stage forward.

    ``tail="hold"`` repeats the last stage; ``tail="zero"`` sets it to zero.
    """
    if tail not in ("hold", "zero"):
        raise ValueError("tail must be 'hold' or 'zero'")
    out = v.clone()
    n = v.numel()
    for start, size, count in segments:
        start, size, count = int(start), int(size), int(count)
        if size <= 0 or count <= 0:
            continue
        end = start + size * count
        if start < 0 or end > n:
            raise ValueError(f"segment {(start, size, count)} outside a vector of length {n}")
        src = v[start:end].view(count, size)
        dst = out[start:end].view(count, size)
        if count > 1:
            dst[:-1] = src[1:]
        if tail == "zero":
            dst[-1].zero_()
    return out


def check_segments(segments: Sequence[Segment], n: int, what: str):
    for s in segments:
        if len(s) != 3:
            raise ValueError(f"{what} segment {s} is not (start, size, count)")
        if s[0] < 0 or s[0] + s[1] * s[2] > n:
            raise ValueError(f"{what} segment {s} outside length {n}")
