"""Tests for src/tau2/environment/rag.py and RAGToolKit.

Chunking tests require no API key.
ChromaPolicyIndex and RAGToolKit tests use a fake embedding function
so they also run without an API key.
Rate-limit tests control time via a fake clock and never hit the real API.
"""

import math
import random
import threading
from datetime import date
from unittest.mock import MagicMock

import pytest

from tau2.environment.rag import (
    ChromaPolicyIndex,
    ChunkingStrategy,
    chunk_by_fixed,
    chunk_by_headers,
    chunk_by_sentence_window,
    get_chunks,
)
from tau2.environment.toolkit import RAGToolKit, ToolType


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

HEADER_POLICY = """\
# Domain Policy

## Section 1: Returns
You can return any item within 30 days of purchase with a receipt.
Items must be in original condition.

## Section 2: Cancellations
Orders can be cancelled within 24 hours of placement.
After 24 hours, cancellations are subject to a 10% fee.

## Section 3: Shipping
Standard shipping takes 3-5 business days.
Express shipping is available for an extra fee.
"""

PLAIN_TEXT = " ".join(["word"] * 500)  # 500 words, no headers or punctuation


# ---------------------------------------------------------------------------
# chunk_by_headers
# ---------------------------------------------------------------------------


def test_chunk_by_headers_splits_on_h1_and_h2():
    chunks = chunk_by_headers(HEADER_POLICY)
    assert len(chunks) == 4  # intro + 3 sections
    assert any("Returns" in c for c in chunks)
    assert any("Cancellations" in c for c in chunks)
    assert any("Shipping" in c for c in chunks)


def test_chunk_by_headers_no_empty_chunks():
    chunks = chunk_by_headers(HEADER_POLICY)
    assert all(c.strip() for c in chunks)


def test_chunk_by_headers_single_section():
    text = "## Only Section\nSome content here."
    chunks = chunk_by_headers(text)
    assert len(chunks) == 1
    assert "Only Section" in chunks[0]


def test_chunk_by_headers_no_headers_returns_whole_text():
    text = "No headings here at all."
    chunks = chunk_by_headers(text)
    assert len(chunks) == 1
    assert chunks[0] == text


# ---------------------------------------------------------------------------
# chunk_by_fixed
# ---------------------------------------------------------------------------


def test_chunk_by_fixed_200_splits_500_words_into_3():
    chunks = chunk_by_fixed(PLAIN_TEXT, words_per_chunk=200)
    # 500 words / 200 = 2 full chunks + 1 remainder
    assert len(chunks) == 3


def test_chunk_by_fixed_chunk_size():
    chunks = chunk_by_fixed(PLAIN_TEXT, words_per_chunk=200)
    for chunk in chunks[:-1]:  # all but last should be full
        assert len(chunk.split()) == 200


def test_chunk_by_fixed_400_splits_500_words_into_2():
    chunks = chunk_by_fixed(PLAIN_TEXT, words_per_chunk=400)
    assert len(chunks) == 2


def test_chunk_by_fixed_no_empty_chunks():
    chunks = chunk_by_fixed(PLAIN_TEXT, words_per_chunk=200)
    assert all(c.strip() for c in chunks)


def test_chunk_by_fixed_short_text():
    text = "only three words"
    chunks = chunk_by_fixed(text, words_per_chunk=200)
    assert len(chunks) == 1
    assert chunks[0] == text


# ---------------------------------------------------------------------------
# chunk_by_sentence_window
# ---------------------------------------------------------------------------

SENTENCES = "First. Second. Third. Fourth. Fifth. Sixth."


def test_chunk_by_sentence_window_returns_overlapping_chunks():
    chunks = chunk_by_sentence_window(SENTENCES, window=3)
    assert len(chunks) > 1
    # Consecutive chunks should share sentences (overlap)
    words_0 = set(chunks[0].split())
    words_1 = set(chunks[1].split())
    assert words_0 & words_1, "Expected overlapping content between consecutive chunks"


def test_chunk_by_sentence_window_no_empty_chunks():
    chunks = chunk_by_sentence_window(SENTENCES, window=3)
    assert all(c.strip() for c in chunks)


def test_chunk_by_sentence_window_single_sentence():
    chunks = chunk_by_sentence_window("Just one sentence.", window=3)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# get_chunks dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["headers", "fixed_200", "fixed_400", "sentence_window"])
def test_get_chunks_all_strategies_return_list(strategy: ChunkingStrategy):
    chunks = get_chunks(HEADER_POLICY, strategy)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_get_chunks_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_chunks(HEADER_POLICY, "bad_strategy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ChromaPolicyIndex (uses fake embeddings — no API key required)
# ---------------------------------------------------------------------------

def _make_fake_embed_fn(dim: int = 8):
    """Return a deterministic fake embedding function for testing."""
    def fake_embed(texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            rng = random.Random(hash(text) & 0xFFFFFFFF)
            vec = [rng.gauss(0, 1) for _ in range(dim)]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            results.append([v / norm for v in vec])
        return results
    return fake_embed


SIMPLE_POLICY = """\
## Returns
Items can be returned within 30 days.

## Shipping
Standard shipping is 3-5 days.
"""


def test_chroma_policy_index_creates_with_fake_embed():
    index = ChromaPolicyIndex(SIMPLE_POLICY, strategy="headers", _embed_fn=_make_fake_embed_fn())
    assert len(index.chunks) == 2


def test_chroma_policy_index_search_returns_string(tmp_path):
    index = ChromaPolicyIndex(SIMPLE_POLICY, strategy="headers", _embed_fn=_make_fake_embed_fn())
    result = index.search("how do I return a product?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_chroma_policy_index_search_k_limits_results():
    policy = "\n\n".join([f"## Section {i}\nContent {i}." for i in range(6)])
    index = ChromaPolicyIndex(policy, strategy="headers", _embed_fn=_make_fake_embed_fn())
    result_k1 = index.search("content", k=1)
    result_k3 = index.search("content", k=3)
    # k=3 result should be longer (more chunks separated by ---)
    assert result_k3.count("---") >= result_k1.count("---")


def test_chroma_policy_index_empty_policy_raises():
    with pytest.raises(ValueError, match="zero chunks"):
        ChromaPolicyIndex("", strategy="headers", _embed_fn=_make_fake_embed_fn())


def test_chroma_policy_index_different_strategies_produce_different_chunks():
    chunks_h = ChromaPolicyIndex(SIMPLE_POLICY, strategy="headers", _embed_fn=_make_fake_embed_fn()).chunks
    chunks_f = ChromaPolicyIndex(SIMPLE_POLICY, strategy="fixed_200", _embed_fn=_make_fake_embed_fn()).chunks
    # headers splits on ##; fixed_200 merges into word-count windows — they differ
    assert chunks_h != chunks_f


# ---------------------------------------------------------------------------
# RAGToolKit
# ---------------------------------------------------------------------------

def test_ragtoolkit_has_retrieve_policy_tool():
    kit = RAGToolKit()
    assert "retrieve_policy" in kit.tools


def test_ragtoolkit_retrieve_policy_tool_type():
    kit = RAGToolKit()
    assert kit.tool_type("retrieve_policy") == ToolType.READ


def test_ragtoolkit_retrieve_policy_without_index_returns_message():
    kit = RAGToolKit(policy_index=None)
    result = kit.retrieve_policy(query="something")
    assert "not available" in result.lower()


def test_ragtoolkit_retrieve_policy_with_index_returns_text():
    index = ChromaPolicyIndex(SIMPLE_POLICY, strategy="headers", _embed_fn=_make_fake_embed_fn())
    kit = RAGToolKit(policy_index=index)
    result = kit.retrieve_policy(query="return policy")
    assert isinstance(result, str)
    assert len(result) > 0


def test_ragtoolkit_also_has_think_tool():
    """RAGToolKit inherits from GenericToolKit so think should be present."""
    kit = RAGToolKit()
    assert "think" in kit.tools
    assert kit.tool_type("think") == ToolType.THINK


def test_ragtoolkit_subclass_preserves_domain_tools():
    """A domain subclass retains its own tools alongside retrieve_policy and think."""

    from tau2.environment.toolkit import is_tool

    class SampleToolKit(RAGToolKit):
        def __init__(self, db=None, policy_index=None):
            super().__init__(db, policy_index=policy_index)

        @is_tool(ToolType.READ)
        def get_status(self) -> str:
            """Return ok."""
            return "ok"

    kit = SampleToolKit()
    assert "get_status" in kit.tools
    assert "retrieve_policy" in kit.tools
    assert "think" in kit.tools


# ---------------------------------------------------------------------------
# Rate-limit helpers — _is_embed_rate_limit_error
# ---------------------------------------------------------------------------

from tau2.environment.rag import _is_embed_rate_limit_error, _parse_embed_retry_delay


def test_is_rate_limit_error_detects_429():
    assert _is_embed_rate_limit_error(Exception("HTTP 429 Too Many Requests"))


def test_is_rate_limit_error_detects_resource_exhausted():
    assert _is_embed_rate_limit_error(Exception("RESOURCE_EXHAUSTED quota exceeded"))


def test_is_rate_limit_error_detects_quota():
    assert _is_embed_rate_limit_error(Exception("quota exceeded for this project"))


def test_is_rate_limit_error_detects_rate_limit_phrase():
    assert _is_embed_rate_limit_error(Exception("rate limit reached, slow down"))


def test_is_rate_limit_error_ignores_unrelated_errors():
    assert not _is_embed_rate_limit_error(Exception("Connection refused"))
    assert not _is_embed_rate_limit_error(ValueError("invalid argument"))
    assert not _is_embed_rate_limit_error(RuntimeError("model not found"))


# ---------------------------------------------------------------------------
# Rate-limit helpers — _parse_embed_retry_delay
# ---------------------------------------------------------------------------

def test_parse_retry_delay_from_message_float():
    err = Exception("Please retry in 40.726638739s.")
    assert _parse_embed_retry_delay(err) == pytest.approx(40.726638739)


def test_parse_retry_delay_from_message_integer():
    err = Exception("Retry in 10s after backoff")
    assert _parse_embed_retry_delay(err) == pytest.approx(10.0)


def test_parse_retry_delay_case_insensitive():
    err = Exception("RETRY IN 5.5S please")
    assert _parse_embed_retry_delay(err) == pytest.approx(5.5)


def test_parse_retry_delay_returns_none_when_no_hint():
    err = Exception("Something went wrong")
    assert _parse_embed_retry_delay(err) is None


def test_parse_retry_delay_from_body_dict():
    """Error objects that carry a .body dict with retryDelay are parsed correctly."""
    err = Exception("quota")
    err.body = {  # type: ignore[attr-defined]
        "error": {
            "details": [
                {"@type": "google.rpc.RetryInfo", "retryDelay": "40s"},
            ]
        }
    }
    assert _parse_embed_retry_delay(err) == pytest.approx(40.0)


def test_parse_retry_delay_body_takes_precedence_after_message_match():
    """Message-based match is found first (regex runs before body parse)."""
    err = Exception("retry in 25.0s")
    err.body = {"error": {"details": [{"retryDelay": "99s"}]}}  # type: ignore[attr-defined]
    # regex in message matches first → 25.0, not 99
    assert _parse_embed_retry_delay(err) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Fake clock for rate-limiter tests
# ---------------------------------------------------------------------------

class _FakeTime:
    """Drop-in replacement for the `time` module inside rag.py."""

    def __init__(self, start: float = 0.0):
        self._now = start
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self._now

    def sleep(self, secs: float) -> None:
        self.sleep_calls.append(secs)
        self._now += secs


@pytest.fixture()
def embed_state(monkeypatch):
    """Reset module-level embed rate-limiter state; yield the rag module."""
    import tau2.environment.rag as rag_mod

    orig_window = list(rag_mod._embed_window)
    orig_count = rag_mod._embed_daily_count
    orig_date = rag_mod._embed_daily_date

    rag_mod._embed_window.clear()
    rag_mod._embed_daily_count = 0
    rag_mod._embed_daily_date = date.today().isoformat()

    yield rag_mod

    rag_mod._embed_window.clear()
    rag_mod._embed_window.extend(orig_window)
    rag_mod._embed_daily_count = orig_count
    rag_mod._embed_daily_date = orig_date


# ---------------------------------------------------------------------------
# _embed_rate_limit_acquire — RPM enforcement
# ---------------------------------------------------------------------------

from tau2.environment.rag import _embed_rate_limit_acquire


def test_rpm_acquire_allows_calls_under_limit(monkeypatch, embed_state):
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 3)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 1000)

    for _ in range(3):
        rag_mod._embed_rate_limit_acquire()

    assert clock.sleep_calls == [], "No sleep should occur when under the RPM limit"
    assert len(rag_mod._embed_window) == 3
    assert rag_mod._embed_daily_count == 3


def test_rpm_acquire_sleeps_when_window_full(monkeypatch, embed_state):
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 2)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 1000)

    # Fill the window
    rag_mod._embed_rate_limit_acquire()
    rag_mod._embed_rate_limit_acquire()
    assert clock.sleep_calls == []

    # Third call must sleep until the oldest entry expires
    rag_mod._embed_rate_limit_acquire()
    assert len(clock.sleep_calls) == 1
    assert clock.sleep_calls[0] == pytest.approx(60.0)


def test_rpm_acquire_resumes_after_window_clears(monkeypatch, embed_state):
    """Multiple overflow cycles each trigger exactly one sleep until the window clears."""
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 2)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 1000)

    # Sequence with limit=2, 5 calls:
    #   calls 1-2 → fill window at t=0, no sleep
    #   call  3   → overflow → sleep 60s → clock=60 → prune [0,0] → add at 60
    #   call  4   → window=[60], len=1 < 2 → no sleep, add → [60,60]
    #   call  5   → overflow → sleep 60s → clock=120 → prune [60,60] → add at 120
    for _ in range(5):
        rag_mod._embed_rate_limit_acquire()

    assert len(clock.sleep_calls) == 2
    assert all(s == pytest.approx(60.0) for s in clock.sleep_calls)
    assert rag_mod._embed_daily_count == 5


def test_rpm_window_size_never_exceeds_limit(monkeypatch, embed_state):
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 3)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 1000)

    for _ in range(10):
        rag_mod._embed_rate_limit_acquire()
        assert len(rag_mod._embed_window) <= 3


# ---------------------------------------------------------------------------
# _embed_rate_limit_acquire — daily cap enforcement
# ---------------------------------------------------------------------------

def test_daily_limit_raises_after_exhaustion(monkeypatch, embed_state):
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 1000)  # don't trigger RPM
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 3)

    for _ in range(3):
        rag_mod._embed_rate_limit_acquire()

    with pytest.raises(RuntimeError, match="daily limit"):
        rag_mod._embed_rate_limit_acquire()


def test_daily_limit_raises_before_rpm_check(monkeypatch, embed_state):
    """Daily exhaustion must raise even when there is room in the RPM window."""
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 50)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 2)

    rag_mod._embed_rate_limit_acquire()
    rag_mod._embed_rate_limit_acquire()

    with pytest.raises(RuntimeError, match="daily limit"):
        rag_mod._embed_rate_limit_acquire()

    assert clock.sleep_calls == [], "No RPM sleep should occur before daily error"


def test_daily_counter_resets_on_new_day(monkeypatch, embed_state):
    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 1000)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 60.0)
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 2)

    rag_mod._embed_rate_limit_acquire()
    rag_mod._embed_rate_limit_acquire()

    # Simulate a new calendar day by changing the stored date to yesterday
    rag_mod._embed_daily_date = "1970-01-01"

    # Should not raise — daily counter resets
    rag_mod._embed_rate_limit_acquire()
    assert rag_mod._embed_daily_count == 1


# ---------------------------------------------------------------------------
# _embed_rate_limit_acquire — thread safety
# ---------------------------------------------------------------------------

def test_concurrent_acquires_never_exceed_rpm_limit(monkeypatch, embed_state):
    """Concurrent threads must not push the window beyond _EMBED_RPM_LIMIT."""
    rag_mod = embed_state
    rpm_limit = 5
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", rpm_limit)
    monkeypatch.setattr(rag_mod, "_EMBED_WINDOW_SECONDS", 0.05)  # 50 ms — fast real sleep
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 1000)

    max_window_seen: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def acquire_and_snapshot():
        try:
            rag_mod._embed_rate_limit_acquire()
            with lock:
                max_window_seen.append(len(rag_mod._embed_window))
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=acquire_and_snapshot) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Unexpected errors in threads: {errors}"
    assert len(max_window_seen) == 12, "All threads should have completed"
    assert max(max_window_seen) <= rpm_limit


# ---------------------------------------------------------------------------
# embed_fn (closure from _make_gemini_embed_fn) — retry behaviour
# — all tests mock google.genai so no API key is needed
# ---------------------------------------------------------------------------

def _make_mock_genai(side_effect_fn):
    """Build a mock genai module whose embed_content uses side_effect_fn."""
    fake_embedding = MagicMock()
    fake_embedding.values = [0.1, 0.2, 0.3]
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = side_effect_fn

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    return mock_genai, fake_response


@pytest.fixture()
def patched_genai(monkeypatch, embed_state):
    """Fixture: patches google.genai.Client and resets embed state + time."""
    import google.genai as real_genai

    rag_mod = embed_state
    clock = _FakeTime()
    monkeypatch.setattr(rag_mod, "time", clock)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr(rag_mod, "_EMBED_DAILY_LIMIT", 1000)
    monkeypatch.setattr(rag_mod, "_EMBED_RPM_LIMIT", 100)

    # We yield a helper that lets each test inject its own side_effect
    yield rag_mod, real_genai, clock


def test_embed_fn_succeeds_on_first_call(monkeypatch, patched_genai):
    rag_mod, real_genai, clock = patched_genai

    call_count = [0]

    def ok(*a, **kw):
        call_count[0] += 1
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.1, 0.2])]
        return resp

    monkeypatch.setattr(real_genai, "Client", lambda api_key: _client_with(ok))

    from tau2.environment.rag import _make_gemini_embed_fn
    result = _make_gemini_embed_fn()(["hello"])

    assert call_count[0] == 1
    assert result == [[0.1, 0.2]]
    assert clock.sleep_calls == []


def _client_with(side_effect_fn):
    """Helper: create a mock genai.Client whose embed_content uses side_effect_fn."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = side_effect_fn
    return mock_client


def test_embed_fn_retries_on_429_and_eventually_succeeds(monkeypatch, patched_genai):
    rag_mod, real_genai, clock = patched_genai

    call_count = [0]

    def flaky(*a, **kw):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("429 resource_exhausted")
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.5])]
        return resp

    monkeypatch.setattr(real_genai, "Client", lambda api_key: _client_with(flaky))

    from tau2.environment.rag import _make_gemini_embed_fn
    result = _make_gemini_embed_fn()(["text"])

    assert call_count[0] == 3
    assert result == [[0.5]]
    assert len(clock.sleep_calls) == 2  # one sleep per failed attempt


def test_embed_fn_uses_server_retry_delay(monkeypatch, patched_genai):
    rag_mod, real_genai, clock = patched_genai

    def once_then_ok(*a, **kw):
        if not once_then_ok.called:
            once_then_ok.called = True
            err = Exception("Please retry in 30.0s resource_exhausted")
            raise err
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[1.0])]
        return resp

    once_then_ok.called = False

    monkeypatch.setattr(real_genai, "Client", lambda api_key: _client_with(once_then_ok))

    from tau2.environment.rag import _make_gemini_embed_fn
    _make_gemini_embed_fn()(["x"])

    # Server said 30s → we add 1s buffer → sleep should be ~31s
    assert len(clock.sleep_calls) == 1
    assert clock.sleep_calls[0] == pytest.approx(31.0)


def test_embed_fn_uses_exponential_backoff_without_server_hint(monkeypatch, patched_genai):
    rag_mod, real_genai, clock = patched_genai

    call_count = [0]

    def twice_fail(*a, **kw):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("quota exceeded")
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.0])]
        return resp

    monkeypatch.setattr(real_genai, "Client", lambda api_key: _client_with(twice_fail))
    monkeypatch.setattr(rag_mod, "_EMBED_BACKOFF_INITIAL", 5.0)
    monkeypatch.setattr(rag_mod, "_EMBED_BACKOFF_MULTIPLIER", 2.0)

    from tau2.environment.rag import _make_gemini_embed_fn
    _make_gemini_embed_fn()(["x"])

    assert len(clock.sleep_calls) == 2
    assert clock.sleep_calls[0] == pytest.approx(5.0)   # attempt 0: 5 * 2^0
    assert clock.sleep_calls[1] == pytest.approx(10.0)  # attempt 1: 5 * 2^1


def test_embed_fn_reraises_non_rate_limit_error_immediately(monkeypatch, patched_genai):
    rag_mod, real_genai, clock = patched_genai

    def always_fail(*a, **kw):
        raise ValueError("invalid model name")

    monkeypatch.setattr(real_genai, "Client", lambda api_key: _client_with(always_fail))

    from tau2.environment.rag import _make_gemini_embed_fn
    with pytest.raises(ValueError, match="invalid model name"):
        _make_gemini_embed_fn()(["x"])

    assert clock.sleep_calls == [], "Non-rate-limit errors must not trigger backoff sleep"


def test_embed_fn_gives_up_after_max_retries(monkeypatch, patched_genai):
    rag_mod, real_genai, clock = patched_genai

    monkeypatch.setattr(rag_mod, "_EMBED_MAX_RETRIES", 3)

    call_count = [0]

    def always_429(*a, **kw):
        call_count[0] += 1
        raise Exception("429 rate limit")

    monkeypatch.setattr(real_genai, "Client", lambda api_key: _client_with(always_429))

    from tau2.environment.rag import _make_gemini_embed_fn
    with pytest.raises(Exception, match="rate limit"):
        _make_gemini_embed_fn()(["x"])

    # 1 initial + 3 retries = 4 total calls
    assert call_count[0] == 4
    assert len(clock.sleep_calls) == 3  # sleep between each retry, not after final failure


def test_embed_fn_no_api_key_raises_before_any_call(monkeypatch, embed_state):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from tau2.environment.rag import _make_gemini_embed_fn
    with pytest.raises(ValueError, match="No Gemini API key"):
        _make_gemini_embed_fn()
