import pytest
from tau2.domains.enosa_masias.data_model import EnosaDB, User, Supply, Ticket
from tau2.domains.enosa_masias.tools import EnosaToolKit

@pytest.fixture
def db():
    return EnosaDB(
        users={"123": User(user_id="123", name="Test", phone="123", email="t@t.com")},
        supplies={"S-1": Supply(supply_number="S-1", owner_id="123", address="Av A", status="active", debt_amount=0.0)},
        tickets={}
    )

@pytest.fixture
def toolkit(db):
    return EnosaToolKit(db=db)

def test_get_user(toolkit):
    res = toolkit.get_user_details("123")
    assert res["name"] == "Test"

def test_create_ticket(toolkit):
    res = toolkit.create_ticket("123", "power_outage", "No light", "S-1")
    assert "ticket" in res
    assert res["ticket"]["issue_type"] == "power_outage"
    
def test_get_office_locations_found(toolkit):
    res = toolkit.get_office_locations("piura")
    assert "address_and_hours" in res

def test_get_office_locations_not_found(toolkit):
    res = toolkit.get_office_locations("lima")
    assert "error" in res