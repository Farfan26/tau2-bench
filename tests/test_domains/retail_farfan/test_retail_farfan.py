import pytest
<<<<<<< HEAD
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


from tau2.domains.retail_farfan.data_model import (
    User,
    Product,
    Order,
    Return,
    Payment,
    RetailDB,
)
from tau2.domains.retail_farfan.tools import RetailTools
=======
from src.tau2.domains.retail_farfan.environment import get_environment
>>>>>>> cf7efe9 (Entrega 2: Subida de código, tareas y resultados parciales de simulación)


def test_environment_loads():
    # Verifica que el entorno se pueda inicializar
    env = get_environment()
    assert env is not None
    assert env.domain_name == "retail_farfan"
