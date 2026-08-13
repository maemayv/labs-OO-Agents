# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that both async clients apply the endpoint concurrency ceiling.

`max_concurrent` bounds calls in flight to one endpoint. CompletionClient and ResponsesClient
issue their requests through different litellm entry points, so each is covered here; parity
between them is the point.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient, concurrency

ENDPOINT = "https://gateway.test/v1"


@pytest.fixture(autouse=True)
def _clear():
    concurrency.reset()
    yield
    concurrency.reset()


def make_mock_completion_response(content: str = "ok") -> litellm.ModelResponse:
    """A minimal ModelResponse; _collect_async type-checks what the provider returns."""
    msg = litellm.Message(content=content, role="assistant")
    choice = litellm.Choices(message=msg, index=0, finish_reason="stop")
    return litellm.ModelResponse(choices=[choice], model="test-model")


def make_mock_responses_response(content: str = "ok"):
    resp = MagicMock()
    resp.output = [MagicMock(type="message", content=[MagicMock(type="output_text", text=content)])]
    resp.output_text = content
    resp.usage = None
    return resp


def _tracking(peak: dict, response):
    """An AsyncMock that records how many calls are in flight at once."""

    async def _call(*_a, **_kw):
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        try:
            await asyncio.sleep(0.01)
            return response
        finally:
            peak["now"] -= 1

    return AsyncMock(side_effect=_call)


class TestClientInit:
    """`max_concurrent` is stored, not forwarded to litellm."""

    def test_defaults_to_unbounded(self):
        assert CompletionClient(model="test-model").max_concurrent == 0
        assert ResponsesClient(model="test-model").max_concurrent == 0

    @pytest.mark.parametrize("cls", [CompletionClient, ResponsesClient])
    def test_max_concurrent_does_not_leak_into_config(self, cls):
        """Anything left in self.config is passed to litellm, which would reject it."""
        client = cls(model="test-model", api_base=ENDPOINT, max_concurrent=4)
        assert client.max_concurrent == 4
        assert "max_concurrent" not in client.config


class TestAsyncCeiling:
    """acall() holds a slot for the duration of the request."""

    @pytest.mark.asyncio
    async def test_completion_client_bounds_calls_in_flight(self):
        client = CompletionClient(model="test-model", api_base=ENDPOINT, max_concurrent=3)
        peak = {"now": 0, "max": 0}
        with patch("litellm.acompletion", _tracking(peak, make_mock_completion_response("hello"))):
            await asyncio.gather(
                *(client.acall(messages=[{"role": "user", "content": "hi"}]) for _ in range(12))
            )
        assert peak["max"] == 3

    @pytest.mark.asyncio
    async def test_responses_client_bounds_calls_in_flight(self):
        client = ResponsesClient(model="test-model", api_base=ENDPOINT, max_concurrent=2)
        peak = {"now": 0, "max": 0}
        with patch("litellm.aresponses", _tracking(peak, make_mock_responses_response("hello"))):
            await asyncio.gather(
                *(client.acall(messages=[{"role": "user", "content": "hi"}]) for _ in range(10))
            )
        assert peak["max"] == 2

    @pytest.mark.asyncio
    async def test_zero_leaves_calls_unbounded(self):
        """The default must not change existing behaviour."""
        client = CompletionClient(model="test-model", api_base=ENDPOINT, max_concurrent=0)
        peak = {"now": 0, "max": 0}
        with patch("litellm.acompletion", _tracking(peak, make_mock_completion_response())):
            await asyncio.gather(
                *(client.acall(messages=[{"role": "user", "content": "hi"}]) for _ in range(6))
            )
        assert peak["max"] == 6

    @pytest.mark.asyncio
    async def test_two_clients_on_one_endpoint_share_the_ceiling(self):
        """A per-client ceiling would let N clients each take the full limit."""
        a = CompletionClient(model="model-a", api_base=ENDPOINT, max_concurrent=2)
        b = CompletionClient(model="model-b", api_base=ENDPOINT, max_concurrent=2)
        peak = {"now": 0, "max": 0}
        with patch("litellm.acompletion", _tracking(peak, make_mock_completion_response())):
            await asyncio.gather(
                *(a.acall(messages=[{"role": "user", "content": "hi"}]) for _ in range(6)),
                *(b.acall(messages=[{"role": "user", "content": "hi"}]) for _ in range(6)),
            )
        assert peak["max"] == 2
