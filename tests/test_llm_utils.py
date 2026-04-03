import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils import llm_utils
from tau2.utils.llm_utils import generate


class FakeToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, tool_id: str, name: str, arguments: str):
        self.id = tool_id
        self.function = FakeToolFunction(name=name, arguments=arguments)


class FakeResponseMessage:
    def __init__(self, content="ok", tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, content="ok", tool_calls=None):
        self.finish_reason = "stop"
        self.message = FakeResponseMessage(content=content, tool_calls=tool_calls)

    def to_dict(self):
        return {}


class FakeResponse:
    def __init__(self, content="ok", tool_calls=None, usage=None):
        self.choices = [FakeChoice(content=content, tool_calls=tool_calls)]
        self._usage = usage
        self.model = "gpt-4o-mini"

    def get(self, key):
        if key == "usage":
            return self._usage
        return None


class FakeRateLimitError(Exception):
    def __init__(self, message="429 Too Many Requests", status_code=429):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def reset_rate_limiters(monkeypatch):
    monkeypatch.setattr(llm_utils, "_ROLLING_RATE_LIMITERS", {})
    monkeypatch.setattr(llm_utils, "_DAILY_RATE_LIMITERS", {})


def make_offline_generate_stub(responses):
    response_iter = iter(responses)

    def _completion(**kwargs):
        return next(response_iter)

    return _completion


@pytest.fixture
def model() -> str:
    return "gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


def test_generate_no_tool_call(monkeypatch, model: str, messages: list[Message]):
    monkeypatch.setattr(
        llm_utils,
        "completion",
        make_offline_generate_stub([FakeResponse(content="Moonbase Alpha")]),
    )
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content == "Moonbase Alpha"


def test_generate_tool_call(
    monkeypatch, model: str, tool_call_messages: list[Message], tool: Tool
):
    monkeypatch.setattr(
        llm_utils,
        "completion",
        make_offline_generate_stub(
            [
                FakeResponse(
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            tool_id="call_123",
                            name="calculate_square",
                            arguments='{"x": 5}',
                        )
                    ],
                ),
                FakeResponse(content="25"),
            ]
        ),
    )
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"


def test_generate_rate_limit_requests_per_window(monkeypatch, model: str, messages: list[Message]):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    generate(
        model,
        messages,
        rate_limit_requests_per_minute=1,
        rate_limit_window_seconds=10,
    )
    generate(
        model,
        messages,
        rate_limit_requests_per_minute=1,
        rate_limit_window_seconds=10,
    )

    assert fake_clock.sleeps == [10.0]


def test_generate_rate_limit_tokens_per_window(monkeypatch, model: str, messages: list[Message]):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(
        llm_utils,
        "get_response_usage",
        lambda response: {"prompt_tokens": 7, "completion_tokens": 5},
    )
    monkeypatch.setattr(llm_utils, "_estimate_request_tokens", lambda **kwargs: 7)

    generate(
        model,
        messages,
        rate_limit_tokens_per_minute=10,
        rate_limit_window_seconds=10,
    )
    generate(
        model,
        messages,
        rate_limit_tokens_per_minute=10,
        rate_limit_window_seconds=10,
    )

    assert fake_clock.sleeps == [10.0]


def test_generate_shared_bucket_limits_requests_across_callers(
    monkeypatch, model: str, messages: list[Message]
):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    generate(
        model,
        messages,
        rate_limit_requests_per_minute=1,
        rate_limit_window_seconds=10,
        rate_limit_bucket="google-free-tier",
    )
    generate(
        model,
        messages,
        rate_limit_requests_per_minute=1,
        rate_limit_window_seconds=10,
        rate_limit_bucket="google-free-tier",
    )

    assert fake_clock.sleeps == [10.0]


def test_generate_shared_bucket_limits_tokens_across_callers(
    monkeypatch, model: str, messages: list[Message]
):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(
        llm_utils,
        "get_response_usage",
        lambda response: {"prompt_tokens": 7, "completion_tokens": 5},
    )
    monkeypatch.setattr(llm_utils, "_estimate_request_tokens", lambda **kwargs: 7)

    generate(
        model,
        messages,
        rate_limit_tokens_per_minute=10,
        rate_limit_window_seconds=10,
        rate_limit_bucket="google-free-tier",
    )
    generate(
        model,
        messages,
        rate_limit_tokens_per_minute=10,
        rate_limit_window_seconds=10,
        rate_limit_bucket="google-free-tier",
    )

    assert fake_clock.sleeps == [10.0]


def test_generate_different_buckets_do_not_share_limits(
    monkeypatch, model: str, messages: list[Message]
):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    generate(
        model,
        messages,
        rate_limit_requests_per_minute=1,
        rate_limit_window_seconds=10,
        rate_limit_bucket="agent",
    )
    generate(
        model,
        messages,
        rate_limit_requests_per_minute=1,
        rate_limit_window_seconds=10,
        rate_limit_bucket="user",
    )

    assert fake_clock.sleeps == []


def test_generate_raises_when_single_request_exceeds_token_window(
    monkeypatch, model: str, messages: list[Message]
):
    monkeypatch.setattr(llm_utils, "_estimate_request_tokens", lambda **kwargs: 11)

    with pytest.raises(ValueError, match="exceed"):
        generate(
            model,
            messages,
            rate_limit_tokens_per_minute=10,
        )


def test_generate_rate_limit_requests_per_day_resets_at_midnight_pacific(
    monkeypatch, model: str, messages: list[Message]
):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("America/Los_Angeles")
    wall_times = iter(
        [
            datetime(2026, 3, 31, 23, 59, tzinfo=pacific),
            datetime(2026, 3, 31, 23, 59, 30, tzinfo=pacific),
            datetime(2026, 4, 1, 0, 0, 1, tzinfo=pacific),
        ]
    )

    monkeypatch.setattr(llm_utils, "_get_rate_limit_wall_time", lambda timezone: next(wall_times))
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    generate(
        model,
        messages,
        rate_limit_requests_per_day=2,
        rate_limit_bucket="google-free-tier",
    )
    generate(
        model,
        messages,
        rate_limit_requests_per_day=2,
        rate_limit_bucket="google-free-tier",
    )
    generate(
        model,
        messages,
        rate_limit_requests_per_day=2,
        rate_limit_bucket="google-free-tier",
    )


def test_generate_rate_limit_requests_per_day_blocks_after_limit(
    monkeypatch, model: str, messages: list[Message]
):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("America/Los_Angeles")
    fixed_now = datetime(2026, 3, 31, 12, 0, tzinfo=pacific)

    monkeypatch.setattr(llm_utils, "_get_rate_limit_wall_time", lambda timezone: fixed_now)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    generate(
        model,
        messages,
        rate_limit_requests_per_day=1,
        rate_limit_bucket="google-free-tier",
    )

    with pytest.raises(ValueError, match="Daily request limit"):
        generate(
            model,
            messages,
            rate_limit_requests_per_day=1,
            rate_limit_bucket="google-free-tier",
        )


def test_generate_retries_on_provider_429_then_succeeds(
    monkeypatch, model: str, messages: list[Message]
):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    calls = iter([FakeRateLimitError(), FakeResponse(content="ok after retry")])

    def flaky_completion(**kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", flaky_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    response = generate(
        model,
        messages,
        rate_limit_429_max_retries=2,
        rate_limit_429_backoff_initial_seconds=1,
        rate_limit_429_backoff_multiplier=2,
        rate_limit_429_backoff_jitter_seconds=0,
    )

    assert response.content == "ok after retry"
    assert fake_clock.sleeps == [1.0]


def test_generate_raises_after_exhausting_provider_429_retries(
    monkeypatch, model: str, messages: list[Message]
):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds: float):
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()

    def always_rate_limited(**kwargs):
        raise FakeRateLimitError()

    monkeypatch.setattr(llm_utils.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_utils.time, "sleep", fake_clock.sleep)
    monkeypatch.setattr(llm_utils, "completion", always_rate_limited)

    with pytest.raises(FakeRateLimitError):
        generate(
            model,
            messages,
            rate_limit_429_max_retries=2,
            rate_limit_429_backoff_initial_seconds=1,
            rate_limit_429_backoff_multiplier=2,
            rate_limit_429_backoff_jitter_seconds=0,
        )

    assert fake_clock.sleeps == [1.0, 2.0]
