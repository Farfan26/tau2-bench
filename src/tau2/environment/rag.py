"""Minimal ChromaDB-based RAG for policy retrieval in tau2 simulations.

Students use this via RAGToolKit — they do not need to modify this file.
The only choice they make is `strategy` in ChromaPolicyIndex.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Literal

import chromadb

ChunkingStrategy = Literal["headers", "fixed_200", "fixed_400", "sentence_window"]

THINK_INSTRUCTION = """
## How to use think
Before calling any operation that modifies data or before rejecting a request,
use think() to articulate:
  1. What the customer is asking for exactly
  2. What the policy says about this situation (based on what retrieve_policy returned)
  3. Whether you have all required information to proceed
  4. Which tool to call and with what arguments
"""


# ---------------------------------------------------------------------------
# Chunking functions
# ---------------------------------------------------------------------------

def chunk_by_headers(text: str) -> list[str]:
    """Split markdown on ## or # headings; each section becomes one chunk."""
    sections = re.split(r"(?m)^(?=#{1,2} )", text)
    return [s.strip() for s in sections if s.strip()]


def chunk_by_fixed(text: str, words_per_chunk: int = 200) -> list[str]:
    """Split text into chunks of roughly `words_per_chunk` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_by_sentence_window(text: str, window: int = 3) -> list[str]:
    """Sliding window of `window` sentences, step = window // 2."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    step = max(1, window // 2)
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + window])
        if chunk:
            chunks.append(chunk)
    return chunks


def get_chunks(text: str, strategy: ChunkingStrategy) -> list[str]:
    """Dispatch to the right chunking function."""
    if strategy == "headers":
        return chunk_by_headers(text)
    elif strategy == "fixed_200":
        return chunk_by_fixed(text, 200)
    elif strategy == "fixed_400":
        return chunk_by_fixed(text, 400)
    elif strategy == "sentence_window":
        return chunk_by_sentence_window(text)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Valid options: headers, fixed_200, fixed_400, sentence_window"
        )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_EMBED_DIM = 768  # Matryoshka truncation; full model outputs 3072


def _make_gemini_embed_fn(
    model: str = DEFAULT_EMBED_MODEL,
    output_dimensionality: int = DEFAULT_EMBED_DIM,
):
    """Return an embedding function backed by Google AI Studio (google-genai SDK).

    Uses the same model and SDK as embeddings_notebook_simplified.ipynb.
    Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    Works with the Google AI Studio free tier (embeddings rate limit: 1 500 RPM).

    Parameters
    ----------
    model : str
        Gemini embedding model. Default: ``"gemini-embedding-001"``.
    output_dimensionality : int
        Matryoshka truncation dimension. Default: 768 (matches the notebook).
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY. "
            "Get a free key at https://aistudio.google.com."
        )

    client = genai.Client(api_key=api_key)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        resp = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=output_dimensionality,
            ),
        )
        return [list(map(float, e.values)) for e in resp.embeddings]

    return embed_fn


# ---------------------------------------------------------------------------
# ChromaPolicyIndex
# ---------------------------------------------------------------------------

class ChromaPolicyIndex:
    """
    Indexes policy chunks in an in-memory ChromaDB collection.

    By default, embeddings use Google AI Studio's ``gemini-embedding-001`` model
    via the ``google-genai`` SDK — the same model and SDK used in
    ``embeddings_notebook_simplified.ipynb``.  Requires ``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY`` in the environment (Google AI Studio free tier is enough;
    the embeddings rate limit is 1 500 RPM, much higher than the chat models).

    Example::

        index = ChromaPolicyIndex(policy_text, strategy="headers")
        relevant = index.search("can I cancel an order made 3 days ago?", k=3)

    Parameters
    ----------
    policy_text : str
        Full text of the policy document.
    strategy : ChunkingStrategy
        One of ``"headers"``, ``"fixed_200"``, ``"fixed_400"``, ``"sentence_window"``.
    _embed_fn : callable, optional
        Override the embedding function (signature: list[str] -> list[list[float]]).
        Used in tests to avoid real API calls.
    """

    def __init__(
        self,
        policy_text: str,
        strategy: ChunkingStrategy = "headers",
        _embed_fn=None,
    ):
        self.strategy = strategy
        self._embed_fn = _embed_fn or _make_gemini_embed_fn()

        self.chunks = get_chunks(policy_text, strategy)
        if not self.chunks:
            raise ValueError("Policy text produced zero chunks. Check the policy file.")

        self._client = chromadb.Client()
        # Unique name to avoid collisions when running multiple instances in tests
        collection_name = f"policy_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        embeddings = self._embed_fn(self.chunks)
        self._collection.add(
            ids=[f"chunk_{i}" for i in range(len(self.chunks))],
            embeddings=embeddings,
            documents=self.chunks,
        )

    def search(self, query: str, k: int = 3) -> str:
        """Return the top-k most relevant policy chunks as a single string."""
        k = min(k, len(self.chunks))
        emb = self._embed_fn([query])
        result = self._collection.query(query_embeddings=emb, n_results=k)
        return "\n\n---\n\n".join(result["documents"][0])
