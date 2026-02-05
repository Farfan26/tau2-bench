import json
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.gemma_tool_converter import create_gemma_system_prompt_with_tools

litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def is_gemma_model(model: str) -> bool:
    """Check if the model is a Gemma model."""
    model_lower = model.lower()
    return "gemma" in model_lower


def parse_gemma_tool_calls(content: str) -> Optional[list[ToolCall]]:
    """
    Parse tool calls from Gemma's ```tool_code``` blocks.

    Args:
        content: The response content from Gemma

    Returns:
        List of ToolCall objects or None if no tool calls found
    """
    import re
    import uuid

    if not content or "```tool_code" not in content:
        return None

    # Extract content from ```tool_code``` blocks
    pattern = r"```tool_code\s*(.*?)\s*```"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        return None

    tool_calls = []
    for match in matches:
        # Parse function call: function_name(arg1=value1, arg2=value2)
        # Match: function_name followed by (...)
        func_pattern = r"(\w+)\((.*?)\)"
        func_match = re.search(func_pattern, match.strip())

        if not func_match:
            logger.warning(f"Could not parse tool call: {match}")
            continue

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        # Parse arguments
        arguments = {}
        if args_str.strip():
            # Simple parsing: split by comma and parse key=value pairs
            # This is a simplified parser - real implementation might need ast.literal_eval
            try:
                # Use exec to safely parse the arguments in a controlled environment
                local_dict = {}
                exec(f"_args = dict({args_str})", {}, local_dict)
                arguments = local_dict.get("_args", {})
            except Exception as e:
                logger.warning(f"Could not parse arguments '{args_str}': {e}")
                # Try fallback: simple key=value parsing
                for pair in args_str.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        try:
                            # Try to evaluate as Python literal
                            arguments[key] = eval(value)
                        except:
                            arguments[key] = value

        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=func_name,
            arguments=arguments,
        )
        tool_calls.append(tool_call)

    return tool_calls if tool_calls else None


def to_gemma_messages(messages: list[Message]) -> list[dict]:
    """
    Convert Tau2 messages to Gemma-compatible format.

    For Gemma:
    - Assistant tool calls are in ```tool_code``` blocks
    - Tool responses are wrapped in ```tool_output``` blocks as user messages
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            # For Gemma, tool calls go in the content as ```tool_code``` blocks
            content = message.content or ""
            if message.is_tool_call():
                # Convert tool calls to Python function call syntax
                tool_code_blocks = []
                for tc in message.tool_calls:
                    # Format arguments as Python kwargs
                    args_parts = []
                    for key, value in tc.arguments.items():
                        # Use repr() to properly quote strings
                        args_parts.append(f"{key}={repr(value)}")
                    args_str = ", ".join(args_parts)

                    func_call = f"{tc.name}({args_str})"
                    tool_code_blocks.append(f"```tool_code\n{func_call}\n```")

                # Combine content and tool calls
                if content:
                    content = content + "\n\n" + "\n".join(tool_code_blocks)
                else:
                    content = "\n".join(tool_code_blocks)

            litellm_messages.append({
                "role": "assistant",
                "content": content,
            })
        elif isinstance(message, ToolMessage):
            # For Gemma, tool responses are user messages with ```tool_output``` blocks
            tool_output = f"```tool_output\n{message.content}\n```"
            litellm_messages.append({
                "role": "user",
                "content": tool_output,
            })
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    Uses standard OpenAI format.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}

    # Check if this is a Gemma model - needs special handling for tools
    use_gemma_format = is_gemma_model(model)
    openai_tools = [tool.openai_schema for tool in tools] if tools else None

    # For Ollama models, ensure large enough context window to avoid truncation
    if "ollama" in model.lower():
        # Set num_ctx to 8192 if not already set (default is 2048 which truncates conversations)
        if "num_ctx" not in kwargs:
            kwargs["num_ctx"] = 8192
            logger.info(f"Setting num_ctx=8192 for Ollama model {model} to preserve full conversation history")

    if use_gemma_format and openai_tools:
        # For Gemma: convert tools to Python signatures and merge into system message
        logger.info(f"Using Gemma function calling format for {model}")

        # Find and enhance system message with tool definitions
        messages_copy = list(messages)  # Make a copy to avoid modifying original
        system_msg_idx = None
        for i, msg in enumerate(messages_copy):
            if isinstance(msg, SystemMessage):
                system_msg_idx = i
                break

        if system_msg_idx is not None:
            # Enhance existing system message
            original_content = messages_copy[system_msg_idx].content
            enhanced_content = create_gemma_system_prompt_with_tools(
                original_content, openai_tools
            )
            messages_copy[system_msg_idx] = SystemMessage(
                role="system", content=enhanced_content
            )
        else:
            # No system message, create one with tools
            tool_prompt = create_gemma_system_prompt_with_tools("", openai_tools)
            messages_copy.insert(0, SystemMessage(role="system", content=tool_prompt))

        # Convert to Gemma format
        litellm_messages = to_gemma_messages(messages_copy)

        # Don't pass tools parameter for Gemma - tools are in the prompt
        try:
            response = completion(
                model=model,
                messages=litellm_messages,
                **kwargs,
            )
        except Exception as e:
            logger.error(e)
            raise e
    else:
        # Standard OpenAI tool calling format
        litellm_messages = to_litellm_messages(messages)
        if openai_tools and tool_choice is None:
            tool_choice = "auto"

        try:
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=openai_tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        except Exception as e:
            logger.error(e)
            raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content

    # Parse tool calls based on model type
    if use_gemma_format:
        # For Gemma: parse tool calls from ```tool_code``` blocks in content
        tool_calls = parse_gemma_tool_calls(content)

        # Remove tool_code blocks from content if tool calls were found
        if tool_calls and content:
            import re
            content = re.sub(r"```tool_code.*?```", "", content, flags=re.DOTALL).strip()
            # If content is now empty, set to None
            content = content if content else None
    else:
        # Standard OpenAI format: tool calls come from response.message.tool_calls
        tool_calls = response.message.tool_calls or []
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
