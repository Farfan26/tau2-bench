import pytest
from tau2.data_model.message import ToolCall
from tau2.domains.retail_farfan.data_model import RetailFarfanDB, User, Order, UserName, UserAddress
from tau2.domains.retail_farfan.environment import get_environment

@pytest.fixture
def retail_db():
    """Inicializa la DB con datos mínimos para las pruebas de herramientas."""
    return RetailFarfanDB(
        products={},
        users={
            "U1": User(
                user_id="U1",
                name=UserName(first_name="Dany", last_name="Farfan"),
                address=UserAddress(address1="Río Viejo", address2="", city="La Arena", country="Peru", state="Piura", zip="20001"),
                email="dany@mail.com",
                payment_methods={}
            )
        },
        orders={
            "ORD1": Order(
                order_id="ORD1",
                user_id="U1",
                address=UserAddress(address1="Río Viejo", address2="", city="La Arena", country="Peru", state="Piura", zip="20001"),
                items=[],
                status="pending",
                payment_history=[]
            )
        },
        tickets={}
    )

def test_get_customer_profile(retail_db):
    env = get_environment(db=retail_db)
    response = env.use_tool("get_customer_profile", customer_id="U1")
    assert response["user_id"] == "U1"
    assert response["name"]["first_name"] == "Dany"

def test_process_refund_security_fail(retail_db):
    """Prueba que el reembolso falle si no se ha verificado el SMS previamente."""
    env = get_environment(db=retail_db)
    # Intentamos reembolsar sin verificar (debería lanzar un error o retornar error en el response)
    response = env.get_response(ToolCall(
        id="call_1",
        name="process_refund",
        arguments={"order_id": "ORD1", "reason": "no longer needed"}
    ))
    # Dependiendo de la implementación, verificamos que la operación no se realizó
    assert "unverified" in str(response).lower() or response.error

def test_sms_verification_flow(retail_db):
    """Prueba el ciclo completo de envío y validación de SMS."""
    env = get_environment(db=retail_db)
    
    # 1. Enviar código
    env.use_tool("send_verification_sms", customer_id="U1")
    assert env.tools.db.users["U1"].current_sms_code == "1234"
    
    # 2. Verificar código
    response = env.use_tool("verify_sms_code", customer_id="U1", code="1234")
    assert "CONFIRMED" in response
    assert env.tools.db.users["U1"].verified is True