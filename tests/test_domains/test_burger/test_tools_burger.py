import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.burger.data_model import BurgerDB, MenuItem
from tau2.domains.burger.environment import get_environment
from tau2.environment.environment import Environment


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
    return get_environment(burger_db)


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
