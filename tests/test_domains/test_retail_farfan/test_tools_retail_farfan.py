import pytest

from tau2.domains.retail_farfan.data_model import RetailDB
from tau2.domains.retail_farfan.tools import RetailTools


def test_get_user_fail():
    db = RetailDB(users={}, products={}, orders={}, returns={}, payments={})
    tools = RetailTools(db)
    with pytest.raises(Exception):
        tools.get_user_details("X")


def test_create_order_fail():
    db = RetailDB(users={}, products={}, orders={}, returns={}, payments={})
    tools = RetailTools(db)
    with pytest.raises(Exception):
        tools.create_order("U1", ["P1"])
