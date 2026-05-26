import pytest
from tau2.domains.retail_farfan.data_model import (
    User,
    Product,
    Order,
    Return,
    Payment,
    RetailDB,
)
from tau2.domains.retail_farfan.tools import RetailTools


# ==========================================
# FIXTURE: Base de datos de prueba en memoria
# ==========================================
@pytest.fixture
def setup_tools():
    """Crea un entorno de prueba fresco antes de cada test."""
    users = {
        "U1": User(
            user_id="U1",
            nombre="Dany",
            email="dany@mail.com",
            telefono="999",
            direccion="Lima",
            estado="activo",
        ),
        "U3": User(
            user_id="U3",
            nombre="Luis",
            email="luis@mail.com",
            telefono="977",
            direccion="Piura",
            estado="bloqueado",
        ),
    }
    products = {
        "P1": Product(
            product_id="P1",
            nombre="Laptop Gamer",
            categoria="tech",
            precio=3500.0,
            stock=5,
            estado="disponible",
            permite_devolucion=True,
        ),
        "P3": Product(
            product_id="P3",
            nombre="Televisor 55",
            categoria="tech",
            precio=2200.0,
            stock=0,
            estado="disponible",
            permite_devolucion=False,
        ),
    }
    orders = {
        "ORD1": Order(
            order_id="ORD1",
            user_id="U1",
            productos=["P1"],
            total=3500.0,
            estado="pendiente",
        ),
        "ORD2": Order(
            order_id="ORD2",
            user_id="U1",
            productos=["P1"],
            total=3500.0,
            estado="entregado",
        ),
    }
    returns = {}
    payments = {
        "PAY1": Payment(
            payment_id="PAY1",
            order_id="ORD1",
            metodo_pago="credit_card",
            estado="pagado",
        )
    }

    db = RetailDB(
        users=users,
        products=products,
        orders=orders,
        returns=returns,
        payments=payments,
    )
    return RetailTools(db)


# ==========================================
# TESTS UNITARIOS
# ==========================================


def test_get_user_details(setup_tools):
    tools = setup_tools
    # Prueba de éxito
    user = tools.get_user_details("U1")
    assert user.nombre == "Dany"

    # Prueba de error (Usuario no existe)
    with pytest.raises(Exception, match="no existe"):
        tools.get_user_details("U99")


def test_search_products(setup_tools):
    tools = setup_tools
    # Prueba de éxito
    results = tools.search_products("Laptop")
    assert len(results) == 1
    assert results[0].product_id == "P1"


def test_create_order(setup_tools):
    tools = setup_tools
    # Prueba de éxito
    order = tools.create_order("U1", ["P1"])
    assert order.total == 3500.0
    assert tools.db.products["P1"].stock == 4  # El stock debió reducirse en 1

    # Prueba de error (Usuario bloqueado)
    with pytest.raises(Exception, match="Usuario bloqueado"):
        tools.create_order("U3", ["P1"])

    # Prueba de error (Producto sin stock)
    with pytest.raises(Exception, match="sin stock"):
        tools.create_order("U1", ["P3"])


def test_cancel_order(setup_tools):
    tools = setup_tools
    # Prueba de éxito (ORD1 está 'pendiente')
    order = tools.cancel_order("ORD1")
    assert order.estado == "cancelado"

    # Prueba de error (ORD2 está 'entregado', no se puede cancelar)
    with pytest.raises(Exception, match="No se puede cancelar"):
        tools.cancel_order("ORD2")


def test_track_order(setup_tools):
    tools = setup_tools
    # Prueba de éxito
    order = tools.track_order("ORD1")
    assert order.user_id == "U1"

    # Prueba de error
    with pytest.raises(Exception, match="no existe"):
        tools.track_order("ORD-INVALIDO")


def test_request_return(setup_tools):
    tools = setup_tools
    # Prueba de éxito (ORD2 está entregado y P1 permite devolución)
    ret = tools.request_return("ORD2", "Llegó roto")
    assert ret.motivo == "Llegó roto"
    assert ret.estado == "solicitada"

    # Prueba de error (ORD1 NO está entregado)
    with pytest.raises(Exception, match="debe estar entregado"):
        tools.request_return("ORD1", "No me gusta")


def test_process_payment(setup_tools):
    tools = setup_tools
    # Prueba de éxito (ORD2 aún no tiene pago)
    pay = tools.process_payment("ORD2", "paypal")
    assert pay.metodo_pago == "paypal"
    assert pay.estado == "pagado"

    # Prueba de error (ORD1 ya fue pagado en el fixture inicial)
    with pytest.raises(Exception, match="ya fue pagado"):
        tools.process_payment("ORD1", "credit_card")
