import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.utils import llm_utils


class FakeResponseMessage:
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content
        self.tool_calls = None


class FakeChoice:
    def __init__(self, content: str):
        self.finish_reason = "stop"
        self.message = FakeResponseMessage(content)

    def to_dict(self):
        return {}


class FakeResponse:
    def __init__(self, content: str, model: str = "gpt-4o-mini"):
        self.choices = [FakeChoice(content)]
        self.model = model

    def get(self, key):
        return None


@pytest.fixture
def user_instructions() -> str:
    return (
        "You are Mia Li. You want to fly from New York to Seattle on May 20 (one way)."
    )


@pytest.fixture
def bad_user_instructions() -> str:
    return "You are Mia Li. You want to fly from Chicago to San Francisco on May 19 (round trip)."


@pytest.fixture
def first_agent_message() -> AssistantMessage:
    return AssistantMessage(
        content="Hello, how can I help you today?", role="assistant"
    )


@pytest.fixture
def user_simulator(user_instructions: str) -> UserSimulator:
    return UserSimulator(llm="gpt-4o-mini", instructions=user_instructions)


def test_user_simulator(
    monkeypatch, user_simulator: UserSimulator, first_agent_message: AssistantMessage
):
    monkeypatch.setattr(
        llm_utils,
        "completion",
        lambda **kwargs: FakeResponse("I need a flight to Seattle on May 20."),
    )
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    user_state = user_simulator.get_init_state()
    assert user_state is not None
    user_msg, user_state = user_simulator.generate_next_message(
        first_agent_message, user_state
    )
    # Check the response is a user message
    assert isinstance(user_msg, UserMessage)
    # Check the state is updated
    assert user_state is not None
    # Check the messages are of the correct type
    assert isinstance(user_state.messages[0], AssistantMessage)
    assert user_state.messages[0].content == first_agent_message.content
    assert isinstance(user_state.messages[1], UserMessage)


def test_user_simulator_set_state(
    user_simulator: UserSimulator,
):
    user_simulator.get_init_state(
        message_history=[
            UserMessage(content="Hello, can you help me find a flight?", role="user"),
            AssistantMessage(
                content="Hello, I can help you find a flight.", role="assistant"
            ),
        ]
    )


def test_user_simulator_gemma_folds_system_after_role_flip(
    monkeypatch, user_instructions: str, first_agent_message: AssistantMessage
):
    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return FakeResponse(
            "I need a one-way flight to Seattle on May 20.",
            model="gemini/gemma-3-12b-it",
        )

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: None)

    user_simulator = UserSimulator(
        llm="gemini/gemma-3-12b-it",
        instructions=user_instructions,
    )

    user_state = user_simulator.get_init_state()
    user_msg, user_state = user_simulator.generate_next_message(
        first_agent_message, user_state
    )

    assert user_msg.content == "I need a one-way flight to Seattle on May 20."
    assert all(message["role"] != "system" for message in captured["messages"])
    assert captured["messages"][0]["role"] == "user"
    assert user_instructions in captured["messages"][0]["content"]
    assert first_agent_message.content in captured["messages"][0]["content"]
