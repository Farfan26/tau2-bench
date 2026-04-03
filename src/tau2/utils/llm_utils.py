import json
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

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

RATE_LIMIT_REQUESTS_PER_MINUTE = "rate_limit_requests_per_minute"
RATE_LIMIT_REQUESTS_PER_DAY = "rate_limit_requests_per_day"
RATE_LIMIT_TOKENS_PER_MINUTE = "rate_limit_tokens_per_minute"
RATE_LIMIT_BUCKET = "rate_limit_bucket"
RATE_LIMIT_WINDOW_SECONDS = "rate_limit_window_seconds"
RATE_LIMIT_TOKEN_RESERVE = "rate_limit_token_reserve"
RATE_LIMIT_DAY_TIMEZONE = "rate_limit_day_timezone"
RATE_LIMIT_429_MAX_RETRIES = "rate_limit_429_max_retries"
RATE_LIMIT_429_BACKOFF_INITIAL_SECONDS = "rate_limit_429_backoff_initial_seconds"
RATE_LIMIT_429_BACKOFF_MAX_SECONDS = "rate_limit_429_backoff_max_seconds"
RATE_LIMIT_429_BACKOFF_MULTIPLIER = "rate_limit_429_backoff_multiplier"
RATE_LIMIT_429_BACKOFF_JITTER_SECONDS = "rate_limit_429_backoff_jitter_seconds"

DEFAULT_RATE_LIMIT_429_MAX_RETRIES = 7
DEFAULT_RATE_LIMIT_429_BACKOFF_INITIAL_SECONDS = 2.0
DEFAULT_RATE_LIMIT_429_BACKOFF_MAX_SECONDS = 30.0
DEFAULT_RATE_LIMIT_429_BACKOFF_MULTIPLIER = 2.0
DEFAULT_RATE_LIMIT_429_BACKOFF_JITTER_SECONDS = 0.0


@dataclass
class _RateLimitConfig:
    requests_per_window: Optional[int]
    requests_per_day: Optional[int]
    tokens_per_window: Optional[int]
    window_seconds: float
    bucket: Optional[str]
    token_reserve: int
    day_timezone: str
    max_429_retries: int
    backoff_initial_seconds: float
    backoff_max_seconds: float
    backoff_multiplier: float
    backoff_jitter_seconds: float


@dataclass
class _RateLimitEntry:
    timestamp: float
    token_count: int


class _RollingWindowRateLimiter:
    def __init__(
        self,
        requests_per_window: Optional[int],
        tokens_per_window: Optional[int],
        window_seconds: float,
    ):
        self.requests_per_window = requests_per_window
        self.tokens_per_window = tokens_per_window
        self.window_seconds = window_seconds
        self.entries: deque[_RateLimitEntry] = deque()
        self.lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> _RateLimitEntry:
        estimated_tokens = max(0, int(estimated_tokens))
        if (
            self.tokens_per_window is not None
            and estimated_tokens > self.tokens_per_window
        ):
            raise ValueError(
                "Estimated request tokens exceed the configured token rate limit window. "
                f"Request estimate: {estimated_tokens}, limit: {self.tokens_per_window}."
            )
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                request_wait_time = self._get_request_wait(now)
                logger.info(f"Calculated request wait time: {request_wait_time:.2f}s")
                tokens_wait_time = self._get_token_wait(now, estimated_tokens)
                logger.info(f"Calculated tokens wait time: {tokens_wait_time:.2f}s")

                wait_time = max(
                    request_wait_time,
                    tokens_wait_time,
                )
                if wait_time <= 0:
                    entry = _RateLimitEntry(timestamp=now, token_count=estimated_tokens)
                    self.entries.append(entry)
                    return entry
            logger.info(
                f"Rate limit reached; waiting {wait_time:.2f}s before sending the next LLM call"
            )
            time.sleep(wait_time)

    def finalize(self, entry: _RateLimitEntry, actual_tokens: Optional[int]) -> None:
        if actual_tokens is None:
            return
        with self.lock:
            entry.token_count = max(entry.token_count, int(actual_tokens))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.entries and self.entries[0].timestamp <= cutoff:
            self.entries.popleft()

    def _get_request_wait(self, now: float) -> float:
        if self.requests_per_window is None:
            return 0.0
        logger.debug(f"Currently {len(self.entries):d} requests in window of size = {self.window_seconds:.1f} seconds, and maximum request_per_windows = {self.requests_per_window:d}")
        if len(self.entries) < self.requests_per_window:
            return 0.0
        oldest_entry = self.entries[0]
        return max(0.0, oldest_entry.timestamp + self.window_seconds - now)

    def _get_token_wait(self, now: float, estimated_tokens: int) -> float:
        if self.tokens_per_window is None:
            return 0.0

        total_tokens = sum(entry.token_count for entry in self.entries)
        logger.debug(f"Currently {total_tokens:d} tokens in window of size = {self.window_seconds:.1f} seconds, and maximum tokens_per_windows = {self.tokens_per_window:d}")
        if total_tokens + estimated_tokens <= self.tokens_per_window:
            return 0.0

        tokens_to_expire = total_tokens + estimated_tokens - self.tokens_per_window
        expired_tokens = 0
        for entry in self.entries:
            expired_tokens += entry.token_count
            if expired_tokens >= tokens_to_expire:
                return max(0.0, entry.timestamp + self.window_seconds - now)
        return self.window_seconds


class _DailyRateLimiter:
    def __init__(self, requests_per_day: int, timezone_name: str):
        self.requests_per_day = requests_per_day
        self.timezone = ZoneInfo(timezone_name)
        self.current_day: Optional[str] = None
        self.requests_today = 0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = _get_rate_limit_wall_time(self.timezone)
            current_day = now.date().isoformat()
            if self.current_day != current_day:
                self.current_day = current_day
                self.requests_today = 0

            if self.requests_today >= self.requests_per_day:
                raise ValueError(
                    "Daily request limit reached for the configured rate limit bucket. "
                    f"Limit: {self.requests_per_day} requests per day in timezone {self.timezone.key}."
                )

            self.requests_today += 1


_ROLLING_RATE_LIMITERS: dict[str, _RollingWindowRateLimiter] = {}
_DAILY_RATE_LIMITERS: dict[str, _DailyRateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()

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


def _get_rate_limit_wall_time(timezone: ZoneInfo) -> datetime:
    return datetime.now(timezone)


def _extract_rate_limit_config(kwargs: dict[str, Any]) -> Optional[_RateLimitConfig]:
    requests_per_window = kwargs.pop(RATE_LIMIT_REQUESTS_PER_MINUTE, None)
    requests_per_day = kwargs.pop(RATE_LIMIT_REQUESTS_PER_DAY, None)
    tokens_per_window = kwargs.pop(RATE_LIMIT_TOKENS_PER_MINUTE, None)
    bucket = kwargs.pop(RATE_LIMIT_BUCKET, None)
    window_seconds = kwargs.pop(RATE_LIMIT_WINDOW_SECONDS, 60.0)
    token_reserve = kwargs.pop(RATE_LIMIT_TOKEN_RESERVE, None)
    day_timezone = kwargs.pop(RATE_LIMIT_DAY_TIMEZONE, "America/Los_Angeles")
    max_429_retries = kwargs.pop(
        RATE_LIMIT_429_MAX_RETRIES, DEFAULT_RATE_LIMIT_429_MAX_RETRIES
    )
    backoff_initial_seconds = kwargs.pop(
        RATE_LIMIT_429_BACKOFF_INITIAL_SECONDS,
        DEFAULT_RATE_LIMIT_429_BACKOFF_INITIAL_SECONDS,
    )
    backoff_max_seconds = kwargs.pop(
        RATE_LIMIT_429_BACKOFF_MAX_SECONDS,
        DEFAULT_RATE_LIMIT_429_BACKOFF_MAX_SECONDS,
    )
    backoff_multiplier = kwargs.pop(
        RATE_LIMIT_429_BACKOFF_MULTIPLIER,
        DEFAULT_RATE_LIMIT_429_BACKOFF_MULTIPLIER,
    )
    backoff_jitter_seconds = kwargs.pop(
        RATE_LIMIT_429_BACKOFF_JITTER_SECONDS,
        DEFAULT_RATE_LIMIT_429_BACKOFF_JITTER_SECONDS,
    )

    if (
        requests_per_window is None
        and requests_per_day is None
        and tokens_per_window is None
        and max_429_retries <= 0
    ):
        return None

    if requests_per_window is not None and requests_per_window <= 0:
        raise ValueError(f"{RATE_LIMIT_REQUESTS_PER_MINUTE} must be greater than 0")
    if requests_per_day is not None and requests_per_day <= 0:
        raise ValueError(f"{RATE_LIMIT_REQUESTS_PER_DAY} must be greater than 0")
    if tokens_per_window is not None and tokens_per_window <= 0:
        raise ValueError(f"{RATE_LIMIT_TOKENS_PER_MINUTE} must be greater than 0")
    if window_seconds <= 0:
        raise ValueError(f"{RATE_LIMIT_WINDOW_SECONDS} must be greater than 0")
    if max_429_retries < 0:
        raise ValueError(f"{RATE_LIMIT_429_MAX_RETRIES} must be non-negative")
    if backoff_initial_seconds <= 0:
        raise ValueError(
            f"{RATE_LIMIT_429_BACKOFF_INITIAL_SECONDS} must be greater than 0"
        )
    if backoff_max_seconds <= 0:
        raise ValueError(
            f"{RATE_LIMIT_429_BACKOFF_MAX_SECONDS} must be greater than 0"
        )
    if backoff_multiplier < 1:
        raise ValueError(
            f"{RATE_LIMIT_429_BACKOFF_MULTIPLIER} must be greater than or equal to 1"
        )
    if backoff_jitter_seconds < 0:
        raise ValueError(
            f"{RATE_LIMIT_429_BACKOFF_JITTER_SECONDS} must be non-negative"
        )

    if token_reserve is None:
        token_reserve = kwargs.get("max_completion_tokens")
    if token_reserve is None:
        token_reserve = kwargs.get("max_tokens")
    token_reserve = int(token_reserve or 0)
    if token_reserve < 0:
        raise ValueError(f"{RATE_LIMIT_TOKEN_RESERVE} must be non-negative")

    return _RateLimitConfig(
        requests_per_window=requests_per_window,
        requests_per_day=requests_per_day,
        tokens_per_window=tokens_per_window,
        window_seconds=float(window_seconds),
        bucket=bucket,
        token_reserve=token_reserve,
        day_timezone=day_timezone,
        max_429_retries=max_429_retries,
        backoff_initial_seconds=backoff_initial_seconds,
        backoff_max_seconds=backoff_max_seconds,
        backoff_multiplier=backoff_multiplier,
        backoff_jitter_seconds=backoff_jitter_seconds,
    )


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    message = str(error).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def _compute_backoff_seconds(
    attempt: int, rate_limit_config: _RateLimitConfig
) -> float:
    backoff = min(
        rate_limit_config.backoff_max_seconds,
        rate_limit_config.backoff_initial_seconds
        * (rate_limit_config.backoff_multiplier ** attempt),
    )
    if rate_limit_config.backoff_jitter_seconds > 0:
        backoff += random.uniform(0, rate_limit_config.backoff_jitter_seconds)
    return backoff


def _estimate_request_tokens(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
) -> int:
    token_counter = getattr(litellm, "token_counter", None)
    if token_counter is not None:
        try:
            token_count = token_counter(model=model, messages=messages, tools=tools)
            if token_count is not None:
                return max(1, int(token_count))
        except Exception as e:
            logger.debug(f"Falling back to heuristic token estimate: {e}")

    content_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content_chars += len(content)
    if tools is not None:
        content_chars += len(json.dumps(tools))
    return max(1, content_chars // 4)


def _get_rate_limiter(
    model: str, config: _RateLimitConfig
) -> _RollingWindowRateLimiter:
    bucket = config.bucket or model
    key = (
        f"{bucket}|rpm={config.requests_per_window}|tpm={config.tokens_per_window}|"
        f"window={config.window_seconds}"
    )
    with _RATE_LIMITERS_LOCK:
        limiter = _ROLLING_RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _RollingWindowRateLimiter(
                requests_per_window=config.requests_per_window,
                tokens_per_window=config.tokens_per_window,
                window_seconds=config.window_seconds,
            )
            _ROLLING_RATE_LIMITERS[key] = limiter
        return limiter


def _get_daily_rate_limiter(model: str, config: _RateLimitConfig) -> _DailyRateLimiter:
    bucket = config.bucket or model
    key = f"{bucket}|rpd={config.requests_per_day}|tz={config.day_timezone}"
    with _RATE_LIMITERS_LOCK:
        limiter = _DAILY_RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _DailyRateLimiter(
                requests_per_day=config.requests_per_day,
                timezone_name=config.day_timezone,
            )
            _DAILY_RATE_LIMITERS[key] = limiter
        return limiter


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

    rate_limit_config = _extract_rate_limit_config(kwargs)

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

    def _call_with_rate_limits(
        *,
        litellm_messages: list[dict],
        completion_kwargs: dict[str, Any],
        token_tools: Optional[list[dict]],
    ):
        estimated_tokens = None
        if rate_limit_config is not None and (
            rate_limit_config.requests_per_window is not None
            or rate_limit_config.tokens_per_window is not None
        ):
            estimated_tokens = _estimate_request_tokens(
                model=model,
                messages=litellm_messages,
                tools=token_tools,
            ) + rate_limit_config.token_reserve

        last_error = None
        max_attempts = 1
        if rate_limit_config is not None:
            max_attempts += rate_limit_config.max_429_retries

        for attempt in range(max_attempts):
            limiter = None
            limiter_entry = None
            if rate_limit_config is not None:
                if rate_limit_config.requests_per_day is not None:
                    _get_daily_rate_limiter(model, rate_limit_config).acquire()
                if estimated_tokens is not None:
                    limiter = _get_rate_limiter(model, rate_limit_config)
                    limiter_entry = limiter.acquire(estimated_tokens)

            try:
                response = completion(**completion_kwargs)
                return response, limiter, limiter_entry
            except Exception as e:
                if limiter is not None and limiter_entry is not None:
                    limiter.finalize(limiter_entry, limiter_entry.token_count)

                if (
                    rate_limit_config is None
                    or not _is_rate_limit_error(e)
                    or attempt >= max_attempts - 1
                ):
                    logger.error(e)
                    raise e

                backoff_seconds = _compute_backoff_seconds(attempt, rate_limit_config)
                logger.warning(
                    f"Received rate limit error from provider; retrying in {backoff_seconds:.2f}s "
                    f"(attempt {attempt + 1}/{max_attempts - 1})"
                )
                time.sleep(backoff_seconds)
                last_error = e

        assert last_error is not None
        raise last_error

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
        response, limiter, limiter_entry = _call_with_rate_limits(
            litellm_messages=litellm_messages,
            token_tools=None,
            completion_kwargs=dict(
                model=model,
                messages=litellm_messages,
                **kwargs,
            ),
        )
    else:
        # Standard OpenAI tool calling format
        litellm_messages = to_litellm_messages(messages)
        if openai_tools and tool_choice is None:
            tool_choice = "auto"
        response, limiter, limiter_entry = _call_with_rate_limits(
            litellm_messages=litellm_messages,
            token_tools=openai_tools,
            completion_kwargs=dict(
                model=model,
                messages=litellm_messages,
                tools=openai_tools,
                tool_choice=tool_choice,
                **kwargs,
            ),
        )
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    if limiter is not None and limiter_entry is not None:
        total_tokens = None
        if usage is not None:
            total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
        limiter.finalize(limiter_entry, total_tokens)
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
