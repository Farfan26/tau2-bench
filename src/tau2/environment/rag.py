"""Minimal ChromaDB-based RAG for policy retrieval in tau2 simulations.

Students use this via RAGToolKit — they do not need to modify this file.
The only choice they make is `strategy` in ChromaPolicyIndex.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections import deque
from datetime import date
from typing import Literal

import chromadb
from loguru import logger

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

# Google AI Studio free-tier limits for gemini-embedding-* models
_EMBED_RPM_LIMIT = 100        # requests per minute
_EMBED_DAILY_LIMIT = 1000     # requests per day
_EMBED_WINDOW_SECONDS = 60.0

# Module-level state shared across all ChromaPolicyIndex instances and threads
_embed_window: deque[float] = deque()
_embed_daily_count: int = 0
_embed_daily_date: str | None = None
_embed_lock = threading.Lock()

_EMBED_MAX_RETRIES = 6
_EMBED_BACKOFF_INITIAL = 5.0
_EMBED_BACKOFF_MAX = 120.0
_EMBED_BACKOFF_MULTIPLIER = 2.0


def _embed_rate_limit_acquire() -> None:
    """Block until firing an embed_content call is within free-tier rate limits.

    Enforces:
    - 100 requests per 60-second rolling window (RPM)
    - 1 000 requests per calendar day (daily cap)

    Thread-safe; shared across all ChromaPolicyIndex instances in the process.
    """
    global _embed_daily_count, _embed_daily_date
    while True:
        with _embed_lock:
            now = time.monotonic()
            today = date.today().isoformat()

            if _embed_daily_date != today:
                _embed_daily_date = today
                _embed_daily_count = 0

            if _embed_daily_count >= _EMBED_DAILY_LIMIT:
                raise RuntimeError(
                    f"Gemini embedding daily limit ({_EMBED_DAILY_LIMIT} requests/day) "
                    "exhausted for today. Wait until tomorrow to run more simulations."
                )

            cutoff = now - _EMBED_WINDOW_SECONDS
            while _embed_window and _embed_window[0] <= cutoff:
                _embed_window.popleft()

            if len(_embed_window) < _EMBED_RPM_LIMIT:
                _embed_window.append(now)
                _embed_daily_count += 1
                return

            wait = _embed_window[0] + _EMBED_WINDOW_SECONDS - now

        logger.info(
            f"Embedding RPM limit ({_EMBED_RPM_LIMIT}/min); "
            f"waiting {wait:.1f}s before next embed call"
        )
        time.sleep(wait)


def _is_embed_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(k in message for k in ("429", "rate limit", "resource_exhausted", "quota"))


def _parse_embed_retry_delay(error: Exception) -> float | None:
    """Extract the server-suggested retry delay (seconds) from a 429 error."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1))
    try:
        body = getattr(error, "body", None) or {}
        if callable(getattr(body, "json", None)):
            body = body.json()
        details = body.get("error", body).get("details", []) if isinstance(body, dict) else []
        for detail in details:
            raw = detail.get("retryDelay", "")
            m = re.match(r"(\d+(?:\.\d+)?)s", str(raw))
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None


def _make_gemini_embed_fn(
    model: str = DEFAULT_EMBED_MODEL,
    output_dimensionality: int = DEFAULT_EMBED_DIM,
):
    """Return an embedding function backed by Google AI Studio (google-genai SDK).

    Uses the same model and SDK as embeddings_notebook_simplified.ipynb.
    Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    Works with the Google AI Studio free tier.

    All chunks are sent in a single batched embed_content call to minimise
    request count (important given the 1 000 req/day free-tier daily limit).
    On 429 errors the function waits for the server-suggested retry delay
    (``retryDelay`` in the error response) or falls back to exponential
    back-off, then retries up to ``_EMBED_MAX_RETRIES`` times.

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
        last_error: Exception | None = None
        for attempt in range(_EMBED_MAX_RETRIES + 1):
            _embed_rate_limit_acquire()
            try:
                resp = client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=output_dimensionality,
                    ),
                )
                return [list(map(float, e.values)) for e in resp.embeddings]
            except Exception as e:
                last_error = e
                if not _is_embed_rate_limit_error(e) or attempt >= _EMBED_MAX_RETRIES:
                    raise

                suggested = _parse_embed_retry_delay(e)
                if suggested is not None:
                    wait = suggested + 1.0  # small buffer on top of server hint
                else:
                    wait = min(
                        _EMBED_BACKOFF_MAX,
                        _EMBED_BACKOFF_INITIAL * (_EMBED_BACKOFF_MULTIPLIER ** attempt),
                    )
                logger.warning(
                    f"Embedding 429 rate-limit; waiting {wait:.1f}s "
                    f"(attempt {attempt + 1}/{_EMBED_MAX_RETRIES})"
                )
                time.sleep(wait)

        assert last_error is not None
        raise last_error

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
