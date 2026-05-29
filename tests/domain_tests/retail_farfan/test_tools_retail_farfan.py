import pytest  # type: ignore

from tau2.domains.retail_farfan.data_model import (
    User,
    Product,
    Order,
    Return,
    Payment,
    RetailDB,
)
from tau2.domains.retail_farfan.tools import RetailTools

# ============================================================
# FIXTURE
# ============================================================


@pytest.fixture
def db():
    users = {
        "U1": User(
            user_id="U1",
            nombre="Dany",
            email="dany@mail.com",
            telefono="999111222",
            direccion="Lima",
            estado="activo",
        ),
        "U3": User(
            user_id="U3",
            nombre="Luis",
            email="luis@mail.com",
            telefono="977666555",
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
            estado="activo",
            permite_devolucion=True,
        ),
        "P2": Product(
            product_id="P2",
            nombre="Mouse Logitech",
            categoria="tech",
            precio=80.0,
            stock=10,
            estado="activo",
            permite_devolucion=True,
        ),
        "P3": Product(
            product_id="P3",
            nombre="Televisor 55",
            categoria="tech",
            precio=2200.0,
            stock=0,
            estado="activo",
            permite_devolucion=False,
        ),
        "P5": Product(
            product_id="P5",
            nombre="Tablet Samsung",
            categoria="tech",
            precio=1200.0,
            stock=3,
            estado="descontinuado",
            permite_devolucion=True,
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
            productos=["P2"],
            total=80.0,
            estado="entregado",
        ),
        "ORD5": Order(
            order_id="ORD5",
            user_id="U1",
            productos=["P2"],
            total=80.0,
            estado="enviado",
        ),
    }
    payments = {
        "PAY1": Payment(
            payment_id="PAY1",
            order_id="ORD2",
            metodo_pago="credit_card",
            estado="pagado",
        ),
    }
    return RetailDB(
        users=users,
        products=products,
        orders=orders,
        returns={},
        payments=payments,
        sms_codes={},
    )


@pytest.fixture
def tools(db):
    return RetailTools(db)


# ============================================================
# get_user_details
# ============================================================


def test_get_user_details_exitoso(tools):
    user = tools.get_user_details("U1")
    assert user.user_id == "U1"


def test_get_user_details_no_existe(tools):
    with pytest.raises(Exception):
        tools.get_user_details("X")


# ============================================================
# search_products
# ============================================================


def test_search_products_exitoso(tools):
    results = tools.search_products("mouse")
    assert any("Mouse" in p.nombre for p in results)


def test_search_products_sin_resultados(tools):
    with pytest.raises(Exception):
        tools.search_products("producto_inexistente_xyz")


# ============================================================
# create_order
# ============================================================


def test_create_order_exitoso(tools):
    order = tools.create_order("U1", ["P2"])
    assert order.user_id == "U1"
    assert order.estado == "pendiente"


def test_create_order_usuario_no_existe(tools):
    with pytest.raises(Exception):
        tools.create_order("U99", ["P2"])


def test_create_order_usuario_bloqueado(tools):
    with pytest.raises(Exception):
        tools.create_order("U3", ["P2"])


def test_create_order_sin_stock(tools):
    with pytest.raises(Exception):
        tools.create_order("U1", ["P3"])


def test_create_order_producto_descontinuado(tools):
    with pytest.raises(Exception):
        tools.create_order("U1", ["P5"])


# ============================================================
# cancel_order
# ============================================================


def test_cancel_order_pendiente(tools):
    order = tools.cancel_order("ORD1")
    assert order.estado == "cancelado"


def test_cancel_order_entregado_falla(tools):
    with pytest.raises(Exception):
        tools.cancel_order("ORD2")


def test_cancel_order_no_existe(tools):
    with pytest.raises(Exception):
        tools.cancel_order("ORD99")


# ============================================================
# track_order
# ============================================================


def test_track_order_exitoso(tools):
    order = tools.track_order("ORD1")
    assert order.order_id == "ORD1"


def test_track_order_no_existe(tools):
    with pytest.raises(Exception):
        tools.track_order("ORD99")


# ============================================================
# request_return
# ============================================================


def test_request_return_exitoso(tools):
    ret = tools.request_return("ORD2", "defective")
    assert ret.order_id == "ORD2"
    assert ret.estado == "solicitada"


def test_request_return_pedido_no_entregado(tools):
    with pytest.raises(Exception):
        tools.request_return("ORD1", "defective")


def test_request_return_no_acepta_devolucion(tools, db):
    db.orders["ORD_TV"] = Order(
        order_id="ORD_TV",
        user_id="U1",
        productos=["P3"],
        total=2200.0,
        estado="entregado",
    )
    with pytest.raises(Exception):
        tools.request_return("ORD_TV", "no me gusta")


def test_request_return_pedido_no_existe(tools):
    with pytest.raises(Exception):
        tools.request_return("ORD99", "defective")


# ============================================================
# send_sms_code
# ============================================================


def test_send_sms_code_exitoso(tools, db):
    tools.send_sms_code("U1")
    assert "U1" in db.sms_codes
    assert db.sms_codes["U1"].isdigit()
    assert len(db.sms_codes["U1"]) == 4


def test_send_sms_code_usuario_no_existe(tools):
    with pytest.raises(Exception):
        tools.send_sms_code("U99")


# ============================================================
# process_payment
# ============================================================


def test_process_payment_exitoso(tools):
    payment = tools.process_payment("ORD1", "credit_card")
    assert payment.estado == "pagado"


def test_process_payment_ya_pagado(tools):
    with pytest.raises(Exception):
        tools.process_payment("ORD2", "credit_card")


def test_process_payment_metodo_invalido(tools):
    with pytest.raises(Exception):
        tools.process_payment("ORD1", "bitcoin")


def test_process_payment_pedido_no_existe(tools):
    with pytest.raises(Exception):
        tools.process_payment("ORD99", "credit_card")
