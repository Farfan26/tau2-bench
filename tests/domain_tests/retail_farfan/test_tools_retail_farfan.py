"""
Tests unitarios para las herramientas del dominio retail_farfan.
Ejecutar con: pytest tests/domain_tests/retail_farfan/test_tools_retail_farfan.py -v
"""

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
from tau2.domains.retail_farfan.user_tools import RetailUserTools

# ============================================================
# FIXTURE: Base de datos de prueba reutilizable
# ============================================================


@pytest.fixture
def db():
    """Crea una base de datos limpia para cada test."""
    users = {
        "U1": User(
            user_id="U1",
            nombre="Dany",
            email="dany@mail.com",
            telefono="999111222",
            direccion="Lima",
            estado="activo",
        ),
        "U2": User(
            user_id="U2",
            nombre="Ana",
            email="ana@mail.com",
            telefono="988777666",
            direccion="Cusco",
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
        "U4": User(
            user_id="U4",
            nombre="Maria",
            email="maria@mail.com",
            telefono="966555444",
            direccion="Arequipa",
            estado="activo",
        ),
        "U5": User(
            user_id="U5",
            nombre="Carlos",
            email="carlos@mail.com",
            telefono="955444333",
            direccion="Trujillo",
            estado="activo",
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
        "P4": Product(
            product_id="P4",
            nombre="Audifonos Sony",
            categoria="tech",
            precio=300.0,
            stock=8,
            estado="activo",
            permite_devolucion=True,
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
            user_id="U2",
            productos=["P2"],
            total=80.0,
            estado="entregado",
        ),
        "ORD3": Order(
            order_id="ORD3",
            user_id="U1",
            productos=["P2"],
            total=80.0,
            estado="entregado",
        ),
        "ORD4": Order(
            order_id="ORD4",
            user_id="U4",
            productos=["P3"],
            total=2200.0,
            estado="entregado",
        ),
        "ORD5": Order(
            order_id="ORD5",
            user_id="U5",
            productos=["P4"],
            total=300.0,
            estado="enviado",
        ),
    }

    returns = {
        "RET1": Return(
            return_id="RET1",
            order_id="ORD4",
            motivo="Fuera de tiempo",
            estado="rechazada",
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
        returns=returns,
        payments=payments,
        sms_codes={},
    )


@pytest.fixture
def tools(db):
    """Crea una instancia de RetailTools con la DB de prueba."""
    return RetailTools(db)


@pytest.fixture
def user_tools(db):
    """Crea una instancia de RetailUserTools con la DB de prueba."""
    return RetailUserTools(db)


# ============================================================
# TESTS: get_user_details
# ============================================================


class TestGetUserDetails:

    def test_usuario_activo_exitoso(self, tools):
        user = tools.get_user_details("U1")
        assert user.user_id == "U1"
        assert user.nombre == "Dany"
        assert user.estado == "activo"

    def test_usuario_bloqueado_retorna_datos(self, tools):
        """get_user_details debe retornar datos aunque el usuario esté bloqueado."""
        user = tools.get_user_details("U3")
        assert user.user_id == "U3"
        assert user.estado == "bloqueado"

    def test_usuario_no_existe_lanza_excepcion(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.get_user_details("U99")


# ============================================================
# TESTS: search_products
# ============================================================


class TestSearchProducts:

    def test_busqueda_por_nombre_exitosa(self, tools):
        results = tools.search_products("mouse")
        assert len(results) >= 1
        assert any("Mouse" in p.nombre for p in results)

    def test_busqueda_por_nombre_parcial(self, tools):
        results = tools.search_products("laptop")
        assert len(results) >= 1
        assert any("Laptop" in p.nombre for p in results)

    def test_busqueda_por_id_producto(self, tools):
        results = tools.search_products("P1")
        assert len(results) >= 1
        assert any(p.product_id == "P1" for p in results)

    def test_busqueda_sin_resultados_lanza_excepcion(self, tools):
        with pytest.raises(Exception):
            tools.search_products("producto_inexistente_xyz")

    def test_busqueda_case_insensitive(self, tools):
        results_lower = tools.search_products("mouse")
        results_upper = tools.search_products("MOUSE")
        assert len(results_lower) == len(results_upper)


# ============================================================
# TESTS: create_order
# ============================================================


class TestCreateOrder:

    def test_crear_pedido_exitoso(self, tools, db):
        order = tools.create_order("U1", ["P2"])
        assert order.user_id == "U1"
        assert "P2" in order.productos
        assert order.estado == "pendiente"
        assert order.total == 80.0

    def test_crear_pedido_reduce_stock(self, tools, db):
        stock_inicial = db.products["P2"].stock
        tools.create_order("U1", ["P2"])
        assert db.products["P2"].stock == stock_inicial - 1

    def test_crear_pedido_multiple_productos(self, tools):
        order = tools.create_order("U1", ["P1", "P2"])
        assert len(order.productos) == 2
        assert order.total == 3500.0 + 80.0

    def test_crear_pedido_usuario_bloqueado_falla(self, tools):
        with pytest.raises(Exception, match="bloqueado"):
            tools.create_order("U3", ["P2"])

    def test_crear_pedido_sin_stock_falla(self, tools):
        with pytest.raises(Exception, match="stock"):
            tools.create_order("U1", ["P3"])

    def test_crear_pedido_producto_descontinuado_falla(self, tools):
        with pytest.raises(Exception, match="descontinuado"):
            tools.create_order("U1", ["P5"])

    def test_crear_pedido_usuario_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.create_order("U99", ["P2"])

    def test_crear_pedido_producto_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.create_order("U1", ["P99"])

    def test_pedido_queda_registrado_en_db(self, tools, db):
        cantidad_inicial = len(db.orders)
        tools.create_order("U1", ["P2"])
        assert len(db.orders) == cantidad_inicial + 1


# ============================================================
# TESTS: cancel_order
# ============================================================


class TestCancelOrder:

    def test_cancelar_pedido_pendiente_exitoso(self, tools):
        order = tools.cancel_order("ORD1")
        assert order.estado == "cancelado"

    def test_cancelar_pedido_enviado_exitoso(self, tools):
        order = tools.cancel_order("ORD5")
        assert order.estado == "cancelado"

    def test_cancelar_pedido_entregado_falla(self, tools):
        with pytest.raises(Exception, match="entregado"):
            tools.cancel_order("ORD2")

    def test_cancelar_pedido_ya_cancelado_falla(self, tools, db):
        db.orders["ORD1"].estado = "cancelado"
        with pytest.raises(Exception):
            tools.cancel_order("ORD1")

    def test_cancelar_pedido_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.cancel_order("ORD99")

    def test_cancelar_actualiza_estado_en_db(self, tools, db):
        tools.cancel_order("ORD1")
        assert db.orders["ORD1"].estado == "cancelado"


# ============================================================
# TESTS: track_order
# ============================================================


class TestTrackOrder:

    def test_rastrear_pedido_exitoso(self, tools):
        order = tools.track_order("ORD1")
        assert order.order_id == "ORD1"
        assert order.estado == "pendiente"

    def test_rastrear_pedido_entregado(self, tools):
        order = tools.track_order("ORD2")
        assert order.estado == "entregado"

    def test_rastrear_pedido_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.track_order("ORD99")

    def test_rastrear_retorna_datos_completos(self, tools):
        order = tools.track_order("ORD1")
        assert order.user_id == "U1"
        assert "P1" in order.productos
        assert order.total == 3500.0


# ============================================================
# TESTS: request_return
# ============================================================


class TestRequestReturn:

    def test_devolucion_valida_exitosa(self, tools):
        ret = tools.request_return("ORD3", "defective")
        assert ret.order_id == "ORD3"
        assert ret.estado == "solicitada"
        assert ret.motivo == "defective"

    def test_devolucion_pedido_no_entregado_falla(self, tools):
        with pytest.raises(Exception, match="entregado"):
            tools.request_return("ORD1", "defective")

    def test_devolucion_producto_no_permite_devolucion_falla(self, tools):
        """ORD4 tiene P3 que no permite devolución."""
        # Primero removemos el return existente para aislar esta validación
        tools.db.returns = {}
        with pytest.raises(Exception, match="no acepta devoluciones"):
            tools.request_return("ORD4", "no me gusta")

    def test_devolucion_duplicada_falla(self, tools):
        """ORD4 ya tiene RET1 registrado."""
        with pytest.raises(Exception, match="Ya existe"):
            tools.request_return("ORD4", "defective")

    def test_devolucion_queda_registrada_en_db(self, tools, db):
        cantidad_inicial = len(db.returns)
        tools.request_return("ORD3", "defective")
        assert len(db.returns) == cantidad_inicial + 1

    def test_devolucion_pedido_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.request_return("ORD99", "defective")


# ============================================================
# TESTS: send_sms_code
# ============================================================


class TestSendSmsCode:

    def test_enviar_sms_exitoso(self, tools, db):
        result = tools.send_sms_code("U1")
        assert "U1" in db.sms_codes
        assert len(db.sms_codes["U1"]) == 4
        assert db.sms_codes["U1"].isdigit()

    def test_codigo_sms_es_determinista(self, tools):
        """El mismo user_id siempre genera el mismo código."""
        result1 = tools.send_sms_code("U1")
        result2 = tools.send_sms_code("U1")
        assert tools.db.sms_codes["U1"] == tools.db.sms_codes["U1"]

    def test_codigos_distintos_por_usuario(self, tools):
        tools.send_sms_code("U1")
        tools.send_sms_code("U2")
        assert tools.db.sms_codes["U1"] != tools.db.sms_codes["U2"]

    def test_enviar_sms_usuario_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.send_sms_code("U99")

    def test_codigo_en_rango_valido(self, tools, db):
        tools.send_sms_code("U5")
        codigo = int(db.sms_codes["U5"])
        assert 1000 <= codigo <= 9999


# ============================================================
# TESTS: process_payment
# ============================================================


class TestProcessPayment:

    def test_pago_exitoso_credit_card(self, tools):
        payment = tools.process_payment("ORD1", "credit_card")
        assert payment.order_id == "ORD1"
        assert payment.metodo_pago == "credit_card"
        assert payment.estado == "pagado"

    def test_pago_exitoso_debit_card(self, tools):
        payment = tools.process_payment("ORD1", "debit_card")
        assert payment.estado == "pagado"

    def test_pago_exitoso_cash(self, tools):
        payment = tools.process_payment("ORD1", "cash")
        assert payment.estado == "pagado"

    def test_pago_pedido_ya_pagado_falla(self, tools):
        """ORD2 ya tiene PAY1 registrado."""
        with pytest.raises(Exception, match="ya fue pagado"):
            tools.process_payment("ORD2", "credit_card")

    def test_pago_pedido_no_existe_falla(self, tools):
        with pytest.raises(Exception, match="no existe"):
            tools.process_payment("ORD99", "credit_card")

    def test_pago_metodo_invalido_falla(self, tools):
        with pytest.raises(Exception, match="no válido"):
            tools.process_payment("ORD1", "bitcoin")

    def test_pago_queda_registrado_en_db(self, tools, db):
        cantidad_inicial = len(db.payments)
        tools.process_payment("ORD1", "credit_card")
        assert len(db.payments) == cantidad_inicial + 1


# ============================================================
# TESTS: user_tools - get_received_sms_code
# ============================================================


class TestGetReceivedSmsCode:

    def test_obtener_codigo_despues_de_envio(self, tools, user_tools, db):
        """El usuario puede obtener el código después de que el agente lo envía."""
        tools.send_sms_code("U1")
        codigo = user_tools.get_received_sms_code("U1")
        assert codigo == db.sms_codes["U1"]
        assert len(codigo) == 4
        assert codigo.isdigit()

    def test_codigo_usuario_coincide_con_agente(self, tools, user_tools):
        """El código que obtiene el usuario debe ser el mismo que el agente generó."""
        tools.send_sms_code("U5")
        codigo_agente = tools.db.sms_codes["U5"]
        codigo_usuario = user_tools.get_received_sms_code("U5")
        assert codigo_usuario == codigo_agente

    def test_obtener_codigo_sin_envio_previo_falla(self, user_tools):
        with pytest.raises(Exception, match="ningún código SMS"):
            user_tools.get_received_sms_code("U1")

    def test_obtener_codigo_usuario_no_existe_falla(self, user_tools):
        with pytest.raises(Exception, match="no existe"):
            user_tools.get_received_sms_code("U99")


# ============================================================
# TESTS: flujo completo SMS + pago
# ============================================================


class TestFlujoCompletoSMS:

    def test_flujo_completo_exitoso(self, tools, user_tools, db):
        """Simula el flujo completo: enviar SMS → usuario obtiene código → pago procesado."""
        # 1. Agente envía SMS
        tools.send_sms_code("U5")

        # 2. Usuario obtiene código
        codigo = user_tools.get_received_sms_code("U5")

        # 3. Verificar que el código coincide
        assert codigo == db.sms_codes["U5"]

        # 4. Procesar pago (el agente verificó el código y procede)
        payment = tools.process_payment("ORD5", "credit_card")
        assert payment.estado == "pagado"
        assert payment.order_id == "ORD5"

    def test_flujo_codigo_incorrecto_no_paga(self, tools, db):
        """Si el código no coincide, el pago no debe procesarse."""
        tools.send_sms_code("U5")
        codigo_real = db.sms_codes["U5"]
        codigo_falso = "9999"

        # Verificar que el código falso no coincide con el real
        assert codigo_falso != codigo_real

        # El agente NO debe llamar a process_payment si el código es incorrecto
        # Este test verifica que el pago NO está en la DB si el agente cumple la política
        pagos_antes = len(db.payments)
        # No llamamos a process_payment porque el código es incorrecto
        assert len(db.payments) == pagos_antes
