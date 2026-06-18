"""Tests proving that chunking_strategy, retrieval_k, and use_think
are correctly wired from get_environment() all the way to the tools and prompt.

All tests use _fake_embed to avoid downloading the ONNX model.
"""
import math
import random

import pytest

from tau2.domains.burger.data_model import BurgerDB, MenuItem
from tau2.domains.burger.environment import get_environment
from tau2.environment.rag import THINK_INSTRUCTION, ChromaPolicyIndex, get_chunks
from tau2.environment.toolkit import RAGToolKit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed(texts):
    def _vec(text, dim=8):
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        v = [rng.gauss(0, 1) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
    return [_vec(t) for t in texts]


POLICY_WITH_HEADERS = """\
## Devoluciones
Puedes devolver cualquier artículo en 30 días con recibo original.

## Cancelaciones
Puedes cancelar dentro de las 24 horas sin cargo.

## Envíos
Solo realizamos envíos dentro del país.
"""


@pytest.fixture
def burger_db():
    return BurgerDB(
        menu_items={
            "burger_classic": MenuItem(
                item_id="burger_classic",
                name="Classic Burger",
                price=8.5,
                available=True,
            )
        },
        orders={},
    )


# ---------------------------------------------------------------------------
# 1. chunking_strategy changes the number of chunks indexed
# ---------------------------------------------------------------------------

def test_headers_strategy_produces_one_chunk_per_section():
    chunks = get_chunks(POLICY_WITH_HEADERS, "headers")
    assert len(chunks) == 3
    assert any("Devoluciones" in c for c in chunks)
    assert any("Cancelaciones" in c for c in chunks)
    assert any("Envíos" in c for c in chunks)


def test_fixed_200_strategy_collapses_short_policy():
    # The sample policy is fewer than 200 words, so fixed_200 yields 1 chunk
    chunks = get_chunks(POLICY_WITH_HEADERS, "fixed_200")
    assert len(chunks) == 1


def test_chunking_strategy_reflected_in_index(monkeypatch):
    """get_environment propagates chunking_strategy to ChromaPolicyIndex."""
    import tau2.domains.burger.environment as burger_env_mod

    created_strategies = []
    original_cls = burger_env_mod.ChromaPolicyIndex

    class SpyIndex(original_cls):
        def __init__(self, policy_text, strategy="headers", _embed_fn=None):
            created_strategies.append(strategy)
            super().__init__(policy_text, strategy=strategy, _embed_fn=_fake_embed)

    monkeypatch.setattr(burger_env_mod, "ChromaPolicyIndex", SpyIndex)

    burger_env_mod.get_environment(
        BurgerDB(menu_items={}, orders={}),
        chunking_strategy="fixed_400",
        use_rag=True,
    )
    assert created_strategies == ["fixed_400"]


# ---------------------------------------------------------------------------
# 2. retrieval_k changes how many chunks are returned
# ---------------------------------------------------------------------------

def test_retrieval_k_controls_number_of_results():
    index = ChromaPolicyIndex(
        POLICY_WITH_HEADERS, strategy="headers", _embed_fn=_fake_embed
    )
    result_k1 = index.search("cancel order", k=1)
    result_k3 = index.search("cancel order", k=3)
    # k=3 joins 3 chunks with "---"; k=1 returns just one chunk (no separator)
    assert "---" not in result_k1
    assert result_k3.count("---") == 2


def test_ragtoolkit_stores_retrieval_k():
    index = ChromaPolicyIndex(
        POLICY_WITH_HEADERS, strategy="headers", _embed_fn=_fake_embed
    )
    kit = RAGToolKit(policy_index=index, retrieval_k=5)
    assert kit.retrieval_k == 5


def test_retrieval_k_flows_into_search(monkeypatch):
    """RAGToolKit.retrieve_policy passes self.retrieval_k to index.search."""
    index = ChromaPolicyIndex(
        POLICY_WITH_HEADERS, strategy="headers", _embed_fn=_fake_embed
    )
    captured = {}
    original_search = index.search

    def _spy(query, k):
        captured["k"] = k
        return original_search(query, k=k)

    monkeypatch.setattr(index, "search", _spy)

    kit = RAGToolKit(policy_index=index, retrieval_k=2)
    kit.retrieve_policy("any question")
    assert captured["k"] == 2


def test_retrieval_k_propagates_through_get_environment(monkeypatch, burger_db):
    """get_environment passes retrieval_k into BurgerTools → RAGToolKit."""
    import tau2.domains.burger.environment as burger_env_mod

    original_cls = burger_env_mod.ChromaPolicyIndex

    class FakeIndex(original_cls):
        def __init__(self, policy_text, strategy="headers", _embed_fn=None):
            super().__init__(policy_text, strategy=strategy, _embed_fn=_fake_embed)

    monkeypatch.setattr(burger_env_mod, "ChromaPolicyIndex", FakeIndex)

    env = burger_env_mod.get_environment(burger_db, retrieval_k=7, use_rag=True)
    assert env.tools.retrieval_k == 7


# ---------------------------------------------------------------------------
# 3. use_think appends THINK_INSTRUCTION to the system prompt
# ---------------------------------------------------------------------------

def test_use_think_false_does_not_add_think_instruction(burger_db):
    env = get_environment(burger_db, use_think=False, use_rag=False)
    assert THINK_INSTRUCTION not in env.policy


def test_use_think_true_appends_think_instruction(monkeypatch, burger_db):
    """With use_think=True and use_rag=True, THINK_INSTRUCTION appears in policy."""
    import tau2.domains.burger.environment as burger_env_mod

    original_cls = burger_env_mod.ChromaPolicyIndex

    class FakeIndex(original_cls):
        def __init__(self, policy_text, strategy="headers", _embed_fn=None):
            super().__init__(policy_text, strategy=strategy, _embed_fn=_fake_embed)

    monkeypatch.setattr(burger_env_mod, "ChromaPolicyIndex", FakeIndex)

    env = burger_env_mod.get_environment(burger_db, use_think=True, use_rag=True)
    assert THINK_INSTRUCTION in env.policy


# ---------------------------------------------------------------------------
# 4. RunConfig accepts env_args and CLI parses --env-args JSON
# ---------------------------------------------------------------------------

def test_runconfig_env_args_default_is_empty_dict():
    from tau2.data_model.simulation import RunConfig
    cfg = RunConfig(domain="burger")
    assert cfg.env_args == {}


def test_runconfig_env_args_stores_values():
    from tau2.data_model.simulation import RunConfig
    cfg = RunConfig(
        domain="burger",
        env_args={"chunking_strategy": "fixed_200", "retrieval_k": 5, "use_think": True},
    )
    assert cfg.env_args["chunking_strategy"] == "fixed_200"
    assert cfg.env_args["retrieval_k"] == 5
    assert cfg.env_args["use_think"] is True


def test_cli_env_args_parsed_as_dict():
    """--env-args JSON string is parsed into a dict by argparse."""
    from argparse import ArgumentParser
    from tau2.cli import add_run_args

    parser = ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args([
        "--domain", "burger",
        "--env-args", '{"chunking_strategy": "fixed_400", "retrieval_k": 5, "use_think": true}',
    ])
    assert isinstance(args.env_args, dict)
    assert args.env_args["chunking_strategy"] == "fixed_400"
    assert args.env_args["retrieval_k"] == 5
    assert args.env_args["use_think"] is True


def test_cli_env_args_default_is_empty_dict():
    from argparse import ArgumentParser
    from tau2.cli import add_run_args

    parser = ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(["--domain", "burger"])
    assert args.env_args == {}
