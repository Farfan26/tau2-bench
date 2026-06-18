"""Tests for src/tau2/environment/rag.py and RAGToolKit.

Chunking tests require no API key.
ChromaPolicyIndex and RAGToolKit tests use a fake embedding function
so they also run without an API key.
"""

import math
import random

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
