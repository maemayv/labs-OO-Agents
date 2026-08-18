# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that a command's timeout bounds the whole control read, not each line.

The control channel carries one line per protocol field, so a reader that
renewed its budget on every line would let a command that keeps the channel
busy run for an unbounded total while still reporting the configured timeout.
"""

import asyncio

import pytest

from nooa.tools._bash_session import BashSession


class _DripReader:
    """A control reader that emits a line forever and never the sentinel.

    Each line arrives comfortably inside the per-line budget, so only a total
    deadline can end the read.
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self.lines_emitted = 0

    async def readline(self) -> bytes:
        await asyncio.sleep(self._interval)
        self.lines_emitted += 1
        return b"noise\n"


@pytest.mark.asyncio
async def test_timeout_bounds_the_whole_read_not_each_line():
    session = BashSession()
    reader = _DripReader(0.02)
    session._control_reader = reader
    session._process = None  # skip the recovery branch; this test is about the budget

    loop = asyncio.get_running_loop()
    started = loop.time()
    # Fails rather than hangs if the budget is per line: the drip never stops.
    lines, timed_out = await asyncio.wait_for(
        session._read_control_until("__SENTINEL__", timeout=0.2), timeout=5.0
    )
    elapsed = loop.time() - started

    assert timed_out is True
    assert elapsed < 1.0, f"read ran {elapsed:.2f}s against a 0.2s budget"
    assert reader.lines_emitted > 1, "the drip should have delivered several lines"
    assert lines == ["noise"] * reader.lines_emitted


@pytest.mark.asyncio
async def test_a_prompt_sentinel_still_returns_before_the_deadline():
    """The deadline must not cut short a read that completes normally."""
    session = BashSession()

    class _PromptReader:
        def __init__(self) -> None:
            self._queue = [b"first\n", b"second\n", b"__SENTINEL__\n"]

        async def readline(self) -> bytes:
            return self._queue.pop(0)

    session._control_reader = _PromptReader()
    session._process = None

    lines, timed_out = await session._read_control_until("__SENTINEL__", timeout=5.0)

    assert timed_out is False
    assert lines == ["first", "second"]
