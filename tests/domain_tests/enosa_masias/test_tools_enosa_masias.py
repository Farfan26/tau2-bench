import pytest
from tau2.data_model.message import ToolCall
from tau2.domains.enosa_masias.data_model import EnosaDB, EnosaInfo, User, Supply
from tau2.domains.enosa_masias.environment import get_environment

@pytest.fixture
def enosa_db():
    return EnosaDB(
        enosa_info=EnosaInfo(company_name="ENOSA", city="Piura", emergency_phone="073-284040", office_hours="08:00-17:00"),
        users={"11111111": User(user_id="11111111", full_name="Juan Perez")},
        supplies={"SUP-01": Supply(supply_id="SUP-01", supply_number="S-101", owner_id="11111111", address="Av. Grau", status="active", debt_amount=0.0)},
        tickets={}
    )

def test_get_supply_details(enosa_db):
    env = get_environment(enosa_db)
    response = env.use_tool("get_supply_details", supply_number="S-101")
    assert response["supply_number"] == "S-101"
    assert response["status"] == "active"

def test_create_ticket(enosa_db):
    env = get_environment(enosa_db)
    # Usamos ToolCall para simular el comportamiento del agente
    response = env.get_response(ToolCall(
        id="call_1",
        name="create_ticket",
        arguments={"reporter_id": "11111111", "issue_type": "power_outage", "description": "No hay luz"}
    ))
    assert not response.error
    assert env.tools.db.tickets["T-001"].issue_type == "power_outage"