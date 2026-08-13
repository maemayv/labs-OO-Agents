# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A self-imposed ceiling on concurrent LLM calls to one endpoint.

This is a knob for callers who want their own concurrency bounded, not a match to any limit the
provider advertises. `retry.py` reacts to what the provider rejects; this bounds what the caller
sends in the first place, which is useful when a run's concurrency is an emergent property of how
many workers it happens to be running rather than a number anyone chose.

The limit is per endpoint, not per client. A provider counts calls arriving at its endpoint, so
several clients pointed at one gateway share one ceiling and clients on different providers do
not compete.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# One semaphore per endpoint, keyed rather than held on the client so that clients built
# separately for the same endpoint queue together. An endpoint has one ceiling: the first limit
# requested for it wins, because a semaphore cannot be resized once callers are waiting on it.
_semaphores: dict[str, asyncio.Semaphore] = {}
_limits: dict[str, int] = {}


def limiter(endpoint: str | None, max_concurrent: int) -> asyncio.Semaphore | None:
    """Return the semaphore bounding `endpoint`, or None when no limit applies.

    A limit of 0 means unbounded, which is the default and preserves existing behaviour.
    """
    if max_concurrent < 0:
        raise ValueError(f"max_concurrent must be >= 0 (0 is unbounded), got {max_concurrent}")
    if max_concurrent == 0:
        return None
    key = endpoint or ""
    if key not in _semaphores:
        _semaphores[key] = asyncio.Semaphore(max_concurrent)
        _limits[key] = max_concurrent
        logger.info("llm: capping %s at %d concurrent calls", key or "default", max_concurrent)
    elif _limits[key] != max_concurrent:
        logger.warning(
            "llm: %s is already capped at %d; ignoring max_concurrent=%d requested by a later "
            "client. One endpoint has one ceiling.",
            key or "default",
            _limits[key],
            max_concurrent,
        )
    return _semaphores[key]


def reset() -> None:
    """Drop every semaphore. For tests, and for a process that reconfigures between runs."""
    _semaphores.clear()
    _limits.clear()


@asynccontextmanager
async def slot(
    semaphore: asyncio.Semaphore | None, endpoint: str | None = None
) -> AsyncIterator[None]:
    """Hold a slot on `semaphore` for the duration of a call, or pass through when None.

    Logs before blocking rather than after acquiring, so the line marks a call that had to wait.
    A caller whose concurrency sits far above the ceiling should be able to see that from the log
    without instrumenting anything.
    """
    if semaphore is None:
        yield
        return
    if semaphore.locked():
        task = asyncio.current_task()
        logger.info(
            "llm: %s waiting for a slot on %s",
            task.get_name() if task else "call",
            endpoint or "default",
        )
    async with semaphore:
        yield
