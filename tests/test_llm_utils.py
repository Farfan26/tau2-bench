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
from tau2.utils.llm_utils import generate, is_gemma3_model, to_gemma_messages


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


def test_generate_gemma_tpm_counts_input_tokens_only(
    monkeypatch, messages: list[Message]
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
        "gemma/gemma-3-27b-it",
        messages,
        rate_limit_tokens_per_minute=15,
        rate_limit_window_seconds=10,
    )
    generate(
        "gemma/gemma-3-27b-it",
        messages,
        rate_limit_tokens_per_minute=15,
        rate_limit_window_seconds=10,
    )

    assert fake_clock.sleeps == []


def test_to_gemma_messages_folds_system_message_into_first_user_message():
    gemma_messages = to_gemma_messages(
        [
            SystemMessage(role="system", content="You are a helpful assistant."),
            UserMessage(role="user", content="What is the capital of the moon?"),
        ]
    )

    assert gemma_messages == [
        {
            "role": "user",
            "content": (
                "You are a helpful assistant.\n\n"
                "What is the capital of the moon?"
            ),
        }
    ]


def test_generate_gemma_never_sends_system_role(
    monkeypatch, tool_call_messages: list[Message], tool: Tool
):
    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return FakeResponse(content="25")

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    response = generate("gemini/gemma-3-27b-it", tool_call_messages, tools=[tool])

    assert response.content == "25"
    assert all(message["role"] != "system" for message in captured["messages"])
    assert captured["messages"][0]["role"] == "user"
    assert "You are a helpful assistant." in captured["messages"][0]["content"]
    assert "What is the square of 5?" in captured["messages"][0]["content"]


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


def test_generate_agent_and_user_models_with_different_buckets_get_separate_request_limiters(
    monkeypatch, messages: list[Message]
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

    agent_model = "gemini/gemma-3-27b-it"
    agent_kwargs = {
        "rate_limit_requests_per_minute": 1,
        "rate_limit_requests_per_day": 14_000,
        "rate_limit_tokens_per_minute": 15_000,
        "rate_limit_token_reserve": 750,
        "rate_limit_window_seconds": 10,
        "rate_limit_bucket": "google-free-tier-27b",
    }
    user_model = "gemini/gemma-3-12b-it"
    user_kwargs = {
        "rate_limit_requests_per_minute": 1,
        "rate_limit_requests_per_day": 14_000,
        "rate_limit_tokens_per_minute": 15_000,
        "rate_limit_token_reserve": 750,
        "rate_limit_window_seconds": 10,
        "rate_limit_bucket": "google-free-tier-12b",
    }

    generate(agent_model, messages, **agent_kwargs)
    generate(user_model, messages, **user_kwargs)

    assert fake_clock.sleeps == []
    assert len(llm_utils._ROLLING_RATE_LIMITERS) == 2
    assert len(llm_utils._DAILY_RATE_LIMITERS) == 2

    generate(agent_model, messages, **agent_kwargs)

    assert fake_clock.sleeps == [10.0]


def test_generate_agent_and_user_models_with_different_buckets_get_separate_token_limiters(
    monkeypatch, messages: list[Message]
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

    agent_model = "gemini/gemma-3-27b-it"
    agent_kwargs = {
        "rate_limit_tokens_per_minute": 10,
        "rate_limit_window_seconds": 10,
        "rate_limit_bucket": "google-free-tier-27b",
    }
    user_model = "gemini/gemma-3-12b-it"
    user_kwargs = {
        "rate_limit_tokens_per_minute": 10,
        "rate_limit_window_seconds": 10,
        "rate_limit_bucket": "google-free-tier-12b",
    }

    generate(agent_model, messages, **agent_kwargs)
    generate(user_model, messages, **user_kwargs)

    assert fake_clock.sleeps == []
    assert len(llm_utils._ROLLING_RATE_LIMITERS) == 2

    generate(user_model, messages, **user_kwargs)

    assert fake_clock.sleeps == [10.0]


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


# ---------------------------------------------------------------------------
# is_gemma3_model
# ---------------------------------------------------------------------------

class TestIsGemma3Model:
    def test_gemma3_variants_are_gemma3(self):
        assert is_gemma3_model("gemma-3-27b-it")
        assert is_gemma3_model("gemini/gemma-3-27b-it")
        assert is_gemma3_model("gemma/gemma-3-12b-it")
        assert is_gemma3_model("gemini/gemma-3-4b-it")

    def test_gemma4_variants_are_not_gemma3(self):
        assert not is_gemma3_model("gemma-4-31b-it")
        assert not is_gemma3_model("gemini/gemma-4-31b-it")
        assert not is_gemma3_model("gemini/gemma-4-26b-a4b-it")

    def test_non_gemma_models_are_not_gemma3(self):
        assert not is_gemma3_model("gpt-4o-mini")
        assert not is_gemma3_model("claude-3-5-sonnet-20241022")
        assert not is_gemma3_model("gemini/gemini-1.5-pro")

    def test_case_insensitive(self):
        assert is_gemma3_model("Gemma-3-27B-IT")
        assert not is_gemma3_model("GEMMA-4-31B-IT")


# ---------------------------------------------------------------------------
# Gemma 4 — generate() uses the standard (OpenAI) tool-calling path
# ---------------------------------------------------------------------------

GEMMA4_MODEL = "gemini/gemma-4-31b-it"


@pytest.fixture
def gemma4_messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a burger shop assistant."),
        UserMessage(role="user", content="I want to place an order."),
    ]


@pytest.fixture
def gemma4_tool_call_messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a burger shop assistant."),
        UserMessage(role="user", content="What burgers do you have?"),
    ]


class TestGemma4Generate:
    """Gemma 4 should use the standard OpenAI tool-calling path."""

    def test_text_response_parsed_correctly(
        self, monkeypatch, gemma4_messages: list[Message]
    ):
        monkeypatch.setattr(
            llm_utils,
            "completion",
            make_offline_generate_stub([FakeResponse(content="Hello! How can I help?")]),
        )
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        response = generate(GEMMA4_MODEL, gemma4_messages)

        assert isinstance(response, AssistantMessage)
        assert response.content == "Hello! How can I help?"
        assert response.tool_calls is None

    def test_native_tool_call_parsed_correctly(
        self, monkeypatch, gemma4_tool_call_messages: list[Message], tool: Tool
    ):
        """Simulates Gemma 4 returning a native functionCall (content=None, tool_calls populated)."""
        monkeypatch.setattr(
            llm_utils,
            "completion",
            make_offline_generate_stub(
                [
                    FakeResponse(
                        content=None,
                        tool_calls=[
                            FakeToolCall(
                                tool_id="91fzcby7",
                                name="calculate_square",
                                arguments='{"x": 9}',
                            )
                        ],
                    )
                ]
            ),
        )
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        response = generate(GEMMA4_MODEL, gemma4_tool_call_messages, tools=[tool])

        assert isinstance(response, AssistantMessage)
        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calculate_square"
        assert response.tool_calls[0].id == "91fzcby7"
        assert response.tool_calls[0].arguments == {"x": 9}

    def test_tool_call_with_no_args_parsed_correctly(
        self, monkeypatch, gemma4_tool_call_messages: list[Message], tool: Tool
    ):
        """Gemma 4 `get_menu()` case — empty args dict."""
        monkeypatch.setattr(
            llm_utils,
            "completion",
            make_offline_generate_stub(
                [
                    FakeResponse(
                        content=None,
                        tool_calls=[
                            FakeToolCall(
                                tool_id="91fzcby7",
                                name="calculate_square",
                                arguments="{}",
                            )
                        ],
                    )
                ]
            ),
        )
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        response = generate(GEMMA4_MODEL, gemma4_tool_call_messages, tools=[tool])

        assert response.tool_calls is not None
        assert response.tool_calls[0].arguments == {}

    def test_native_tool_call_passes_validation(
        self, monkeypatch, gemma4_tool_call_messages: list[Message], tool: Tool
    ):
        """AssistantMessage.validate() must not raise for a Gemma 4 tool call response."""
        monkeypatch.setattr(
            llm_utils,
            "completion",
            make_offline_generate_stub(
                [
                    FakeResponse(
                        content=None,
                        tool_calls=[
                            FakeToolCall(
                                tool_id="abc123",
                                name="calculate_square",
                                arguments='{"x": 4}',
                            )
                        ],
                    )
                ]
            ),
        )
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        response = generate(GEMMA4_MODEL, gemma4_tool_call_messages, tools=[tool])

        # Must not raise
        response.validate()

    def test_system_message_sent_as_system_role(
        self, monkeypatch, gemma4_messages: list[Message], tool: Tool
    ):
        """Gemma 4 uses to_litellm_messages(), which keeps system messages as role=system."""
        captured: dict = {}

        def fake_completion(**kwargs):
            captured["messages"] = kwargs["messages"]
            return FakeResponse(content="ok")

        monkeypatch.setattr(llm_utils, "completion", fake_completion)
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        generate(GEMMA4_MODEL, gemma4_messages, tools=[tool])

        roles = [m["role"] for m in captured["messages"]]
        assert "system" in roles
        assert captured["messages"][0]["role"] == "system"
        assert "burger shop assistant" in captured["messages"][0]["content"]

    def test_tools_passed_to_litellm_in_standard_format(
        self, monkeypatch, gemma4_messages: list[Message], tool: Tool
    ):
        """Gemma 4 must pass tools to LiteLLM (not None), so native tool calling works."""
        captured: dict = {}

        def fake_completion(**kwargs):
            captured["tools"] = kwargs.get("tools")
            return FakeResponse(content="ok")

        monkeypatch.setattr(llm_utils, "completion", fake_completion)
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        generate(GEMMA4_MODEL, gemma4_messages, tools=[tool])

        assert captured["tools"] is not None
        assert len(captured["tools"]) == 1
        assert captured["tools"][0]["function"]["name"] == "calculate_square"

    def test_no_tool_code_injection_in_system_prompt(
        self, monkeypatch, gemma4_messages: list[Message], tool: Tool
    ):
        """Gemma 4 must NOT inject Python-signature tool descriptions into the system prompt."""
        captured: dict = {}

        def fake_completion(**kwargs):
            captured["messages"] = kwargs["messages"]
            return FakeResponse(content="ok")

        monkeypatch.setattr(llm_utils, "completion", fake_completion)
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        generate(GEMMA4_MODEL, gemma4_messages, tools=[tool])

        all_content = " ".join(
            m.get("content", "") or "" for m in captured["messages"]
        )
        assert "```tool_code```" not in all_content
        assert "tool_code" not in all_content
        assert "Available Tools" not in all_content

    def test_tpm_counts_input_tokens_only(
        self, monkeypatch, gemma4_messages: list[Message]
    ):
        """Gemma 4 TPM budget uses only prompt tokens, same as Gemma 3."""

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
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(
            llm_utils,
            "get_response_usage",
            lambda r: {"prompt_tokens": 7, "completion_tokens": 5},
        )
        monkeypatch.setattr(llm_utils, "_estimate_request_tokens", lambda **kwargs: 7)

        # TPM limit of 15 — prompt=7, completion=5. If both counted, second call
        # would need to wait (7+5+7=19 > 15). With only prompt tokens: 7+7=14 ≤ 15.
        generate(GEMMA4_MODEL, gemma4_messages, rate_limit_tokens_per_minute=15, rate_limit_window_seconds=10)
        generate(GEMMA4_MODEL, gemma4_messages, rate_limit_tokens_per_minute=15, rate_limit_window_seconds=10)

        assert fake_clock.sleeps == []

    def test_thinking_only_response_falls_back_to_reasoning_content(
        self, monkeypatch, gemma4_messages: list[Message]
    ):
        """When Gemma 4 returns only thinking tokens (no visible text, no tool calls),
        reasoning_content is surfaced as content so validate() doesn't crash."""

        class FakeThinkingMessage:
            role = "assistant"
            content = None
            tool_calls = None
            reasoning_content = "I need to greet the user warmly."

        class FakeThinkingChoice:
            finish_reason = "stop"
            message = FakeThinkingMessage()

            def to_dict(self):
                return {}

        class FakeThinkingResponse:
            choices = [FakeThinkingChoice()]
            model = GEMMA4_MODEL

            def get(self, key):
                return None

        monkeypatch.setattr(
            llm_utils, "completion", lambda **kwargs: FakeThinkingResponse()
        )
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        response = generate(GEMMA4_MODEL, gemma4_messages)

        assert isinstance(response, AssistantMessage)
        assert response.content == "I need to greet the user warmly."
        assert response.tool_calls is None
        response.validate()  # must not raise


# ---------------------------------------------------------------------------
# Gemma 3 — existing text-based tool calling is unchanged
# ---------------------------------------------------------------------------

class TestGemma3Unchanged:
    """Regression: Gemma 3 still uses the text-based ```tool_code``` path."""

    GEMMA3_MODEL = "gemini/gemma-3-27b-it"

    def test_gemma3_still_folds_system_message(
        self, monkeypatch, tool_call_messages: list[Message], tool: Tool
    ):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured["messages"] = kwargs["messages"]
            return FakeResponse(content="25")

        monkeypatch.setattr(llm_utils, "completion", fake_completion)
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        generate(self.GEMMA3_MODEL, tool_call_messages, tools=[tool])

        roles = [m["role"] for m in captured["messages"]]
        assert "system" not in roles
        assert captured["messages"][0]["role"] == "user"
        assert "You are a helpful assistant." in captured["messages"][0]["content"]

    def test_gemma3_does_not_pass_tools_to_litellm(
        self, monkeypatch, tool_call_messages: list[Message], tool: Tool
    ):
        captured: dict = {}

        def fake_completion(**kwargs):
            captured["tools"] = kwargs.get("tools")
            return FakeResponse(content="25")

        monkeypatch.setattr(llm_utils, "completion", fake_completion)
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        generate(self.GEMMA3_MODEL, tool_call_messages, tools=[tool])

        assert captured["tools"] is None

    def test_gemma3_parses_tool_code_blocks(
        self, monkeypatch, tool_call_messages: list[Message], tool: Tool
    ):
        monkeypatch.setattr(
            llm_utils,
            "completion",
            make_offline_generate_stub(
                [FakeResponse(content="```tool_code\ncalculate_square(x=5)\n```")]
            ),
        )
        monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)
        monkeypatch.setattr(llm_utils, "get_response_usage", lambda r: None)

        response = generate(self.GEMMA3_MODEL, tool_call_messages, tools=[tool])

        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "calculate_square"
        assert response.tool_calls[0].arguments == {"x": 5}
        assert response.content is None
