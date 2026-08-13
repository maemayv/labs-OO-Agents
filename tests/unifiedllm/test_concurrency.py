# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging

import pytest

from nooa.unifiedllm import concurrency


@pytest.fixture(autouse=True)
def _clear():
    concurrency.reset()
    yield
    concurrency.reset()


def test_zero_is_unbounded():
    assert concurrency.limiter("https://example/v1", 0) is None


def test_a_negative_ceiling_is_rejected():
    with pytest.raises(ValueError, match="max_concurrent must be >= 0"):
        concurrency.limiter("https://example/v1", -1)


def test_one_endpoint_shares_one_semaphore():
    """Two clients on the same gateway must queue together, or the ceiling is per client."""
    a = concurrency.limiter("https://example/v1", 4)
    b = concurrency.limiter("https://example/v1", 4)
    assert a is b


def test_one_endpoint_has_one_ceiling(caplog):
    """A second limit for the same endpoint is ignored and said out loud.

    A semaphore cannot be resized once callers are waiting on it, so the alternative to ignoring
    the second value is a second semaphore, which makes the effective ceiling their sum.
    """
    import logging

    first = concurrency.limiter("https://example/v1", 2)
    with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.concurrency"):
        second = concurrency.limiter("https://example/v1", 8)

    assert second is first
    assert [r for r in caplog.records if "already capped at 2" in r.getMessage()]


def test_different_endpoints_do_not_compete():
    a = concurrency.limiter("https://one/v1", 4)
    b = concurrency.limiter("https://two/v1", 4)
    assert a is not b


async def test_slot_bounds_calls_in_flight():
    sem = concurrency.limiter("https://example/v1", 3)
    state = {"now": 0, "peak": 0}

    async def call():
        async with concurrency.slot(sem):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            try:
                await asyncio.sleep(0.01)
            finally:
                state["now"] -= 1

    await asyncio.gather(*(call() for _ in range(20)))
    assert state["peak"] == 3


async def test_slot_is_a_passthrough_when_unbounded():
    ran = False
    async with concurrency.slot(None):
        ran = True
    assert ran


async def test_a_failing_call_releases_its_slot():
    sem = concurrency.limiter("https://example/v1", 1)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            async with concurrency.slot(sem):
                raise RuntimeError("boom")
    # A leaked slot would make this hang rather than complete.
    await asyncio.wait_for(_enter_and_exit(sem), timeout=1.0)


async def _enter_and_exit(sem):
    async with concurrency.slot(sem):
        return True


async def test_a_call_that_waits_is_logged(caplog):
    """A caller configured well above the ceiling should see it in the log, not just feel it."""
    sem = concurrency.limiter("https://example/v1", 1)

    async def call():
        async with concurrency.slot(sem, "https://example/v1"):
            await asyncio.sleep(0.01)

    with caplog.at_level(logging.INFO, logger="nooa.unifiedllm.concurrency"):
        await asyncio.gather(*(call() for _ in range(4)))

    assert [r for r in caplog.records if "waiting for a slot" in r.getMessage()]


async def test_an_uncontended_caller_logs_nothing(caplog):
    sem = concurrency.limiter("https://example/v1", 8)
    with caplog.at_level(logging.INFO, logger="nooa.unifiedllm.concurrency"):
        async with concurrency.slot(sem, "https://example/v1"):
            pass
    assert not [r for r in caplog.records if "waiting for a slot" in r.getMessage()]
