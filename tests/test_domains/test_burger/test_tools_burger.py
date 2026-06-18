import math
import random

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.burger.data_model import BurgerDB, MenuItem
from tau2.domains.burger.environment import get_environment
from tau2.domains.burger.tools import BurgerTools
from tau2.environment.environment import Environment
from tau2.environment.rag import ChromaPolicyIndex
from tau2.environment.toolkit import ToolType

# ---------------------------------------------------------------------------
# Fake embedding function — does not require an API key
# ---------------------------------------------------------------------------

SAMPLE_POLICY = """
## Devoluciones
Puedes devolver cualquier artículo en 30 días con recibo.

## Cancelaciones
Puedes cancelar dentro de las 24 horas sin cargo.
"""


def _fake_embed(texts):
    def make_vec(text, dim=8):
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        v = [rng.gauss(0, 1) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
    return [make_vec(t) for t in texts]


@pytest.fixture
def burger_db() -> BurgerDB:
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


@pytest.fixture
def environment(burger_db: BurgerDB) -> Environment:
    return get_environment(burger_db, use_rag=False)


def test_place_order(environment: Environment):
    response = environment.get_response(
        ToolCall(
            id="1",
            name="place_order",
            arguments={
                "customer_name": "Alex Parker",
                "burger_name": "Classic Burger",
                "quantity": 1,
                "pickup_time": "12:30 PM",
            },
        )
    )
    assert not response.error
    order = environment.tools.db.orders["BURGER-001"]
    assert order.customer_name == "Alex Parker"
    assert order.item_name == "Classic Burger"
    assert order.quantity == 1
    assert order.pickup_time == "12:30 PM"
    assert order.total_price == 8.5


def test_place_order_rejects_unknown_burger(environment: Environment):
    response = environment.get_response(
        ToolCall(
            id="2",
            name="place_order",
            arguments={
                "customer_name": "Alex Parker",
                "burger_name": "Impossible Burger",
                "quantity": 1,
                "pickup_time": "12:30 PM",
            },
        )
    )
    assert response.error


# ---------------------------------------------------------------------------
# E4 tests: RAG integration (no API key required — uses _fake_embed)
# ---------------------------------------------------------------------------


def test_retrieve_policy_returns_text():
    """retrieve_policy must return a non-empty string when an index is provided."""
    index = ChromaPolicyIndex(SAMPLE_POLICY, strategy="headers", _embed_fn=_fake_embed)
    kit = BurgerTools(db=None, policy_index=index)
    result = kit.retrieve_policy(query="Can I cancel my order?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_toolkit_has_think_tool():
    """BurgerTools must expose a 'think' tool inherited from RAGToolKit."""
    kit = BurgerTools(db=None)
    assert "think" in kit.tools
    assert kit.tool_type("think") == ToolType.THINK
