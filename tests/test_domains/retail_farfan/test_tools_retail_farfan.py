"""Unit tests for RetailFarfanTools."""

import json

import pytest

from tau2.domains.retail_farfan.data_model import RetailFarfanDB
from tau2.domains.retail_farfan.tools import RetailFarfanTools
from tau2.domains.retail_farfan.utils import RETAIL_FARFAN_DB_PATH


@pytest.fixture
def db() -> RetailFarfanDB:
    return RetailFarfanDB.load(RETAIL_FARFAN_DB_PATH)


@pytest.fixture
def tools(db: RetailFarfanDB) -> RetailFarfanTools:
    return RetailFarfanTools(db)


# ============================================================
# check_account_status / get_customer_profile
# ============================================================

def test_check_account_status_active(tools: RetailFarfanTools):
    result = tools.check_account_status("U1")
    assert "ACTIVE" in result


def test_check_account_status_blocked(tools: RetailFarfanTools):
    result = tools.check_account_status("U3")
    assert "BLOCKED" in result


def test_check_account_status_invalid_user(tools: RetailFarfanTools):
    result = tools.check_account_status("U999")
    assert "Error" in result


def test_get_customer_profile(tools: RetailFarfanTools):
    result = tools.get_customer_profile("U1")
    data = json.loads(result)
    assert data["user_id"] == "U1"
    assert data["is_blocked"] is False
    assert "ORD1" in data["orders"]


# ============================================================
# search_products / get_product_details
# ============================================================

def test_search_products_found(tools: RetailFarfanTools):
    result = tools.search_products("laptop")
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["product_id"] == "P1"


def test_search_products_not_found(tools: RetailFarfanTools):
    result = tools.search_products("inexistente")
    assert "No products found" in result


def test_get_product_details(tools: RetailFarfanTools):
    result = tools.get_product_details("P1")
    data = json.loads(result)
    assert data["product_id"] == "P1"
    assert "V1A" in data["variants"]


# ============================================================
# get_order_details
# ============================================================

def test_get_order_details(tools: RetailFarfanTools):
    result = tools.get_order_details("ORD1")
    data = json.loads(result)
    assert data["order_id"] == "ORD1"
    assert data["status"] == "pending"


def test_get_order_details_invalid(tools: RetailFarfanTools):
    result = tools.get_order_details("ORD999")
    assert "Error" in result


# ============================================================
# create_order
# ============================================================

def test_create_order_success(tools: RetailFarfanTools):
    result = tools.create_order(customer_id="U5", product_id="P2")
    assert "Success" in result
    assert "U5" in tools.db.users["U5"].orders or len(tools.db.users["U5"].orders) >= 1


def test_create_order_blocked_user(tools: RetailFarfanTools):
    result = tools.create_order(customer_id="U3", product_id="P2")
    assert "FAIL" in result
    assert "blocked" in result.lower()


def test_create_order_no_stock(tools: RetailFarfanTools):
    # P3 (Televisor) has no available variants in db.json
    result = tools.create_order(customer_id="U1", product_id="P3")
    assert "FAIL" in result
    assert "stock" in result.lower()


# ============================================================
# cancel_order
# ============================================================

def test_cancel_order_pending_success(tools: RetailFarfanTools):
    result = tools.cancel_order(order_id="ORD1", reason="no longer needed")
    assert "Success" in result
    assert tools.db.orders["ORD1"].status == "cancelled"


def test_cancel_order_delivered_fails(tools: RetailFarfanTools):
    result = tools.cancel_order(order_id="ORD2", reason="no longer needed")
    assert "FAIL" in result
    assert tools.db.orders["ORD2"].status == "delivered"


# ============================================================
# update_order_items
# ============================================================

def test_update_order_items_success(tools: RetailFarfanTools):
    result = tools.update_order_items(order_id="ORD5", product_id="P4")
    assert "Success" in result
    order = tools.db.orders["ORD5"]
    assert len(order.items) == 1
    assert order.items[0].product_id == "P4"
    assert order.status == "pending (item modified)"


def test_update_order_items_not_pending_fails(tools: RetailFarfanTools):
    result = tools.update_order_items(order_id="ORD2", product_id="P4")
    assert "FAIL" in result


# ============================================================
# request_return
# ============================================================

def test_request_return_success(tools: RetailFarfanTools):
    result = tools.request_return(order_id="ORD3", reason="defective product")
    assert "Success" in result
    assert "Return ID" in result
    assert tools.db.orders["ORD3"].status == "cancelled"
    assert tools.db.orders["ORD3"].return_id is not None


def test_request_return_blocked_account_fails(tools: RetailFarfanTools):
    result = tools.request_return(order_id="ORD4", reason="defective product")
    assert "FAIL" in result
    assert tools.db.orders["ORD4"].status == "delivered"


# ============================================================
# SMS verification / process_refund
# ============================================================

def test_send_and_verify_sms(tools: RetailFarfanTools):
    send_result = tools.send_verification_sms("U1")
    assert "sent" in send_result.lower()
    verify_result = tools.verify_sms_code("U1", "1234")
    assert "Success" in verify_result
    assert tools.db.users["U1"].verified is True


def test_verify_sms_wrong_code(tools: RetailFarfanTools):
    tools.send_verification_sms("U1")
    result = tools.verify_sms_code("U1", "0000")
    assert "Error" in result
    assert tools.db.users["U1"].verified is False


def test_process_refund_requires_verification(tools: RetailFarfanTools):
    result = tools.process_refund(order_id="ORD1", reason="no longer needed")
    assert "FAIL" in result
    assert "not verified" in result.lower()


def test_process_refund_success(tools: RetailFarfanTools):
    tools.send_verification_sms("U1")
    tools.verify_sms_code("U1", "1234")
    result = tools.process_refund(order_id="ORD1", reason="no longer needed")
    assert "Success" in result
    assert tools.db.orders["ORD1"].status == "cancelled"


def test_process_refund_blocked_account(tools: RetailFarfanTools):
    # U4 is blocked, ORD4 belongs to U4
    result = tools.process_refund(order_id="ORD4", reason="no longer needed")
    assert "FAIL" in result
    assert "blocked" in result.lower()


# ============================================================
# pay_order
# ============================================================

def test_pay_order_success(tools: RetailFarfanTools):
    tools.send_verification_sms("U5")
    result = tools.pay_order(order_id="ORD5", payment_method_id="PM5", sms_code="1234")
    assert "Success" in result
    assert tools.db.orders["ORD5"].status == "paid"


def test_pay_order_wrong_code_fails(tools: RetailFarfanTools):
    tools.send_verification_sms("U5")
    result = tools.pay_order(order_id="ORD5", payment_method_id="PM5", sms_code="9999")
    assert "FAIL" in result
    assert tools.db.orders["ORD5"].status == "pending"


def test_pay_order_without_sms_fails(tools: RetailFarfanTools):
    result = tools.pay_order(order_id="ORD5", payment_method_id="PM5", sms_code="1234")
    assert "FAIL" in result


# ============================================================
# calculate / transfer_to_human_agents
# ============================================================

def test_calculate_valid(tools: RetailFarfanTools):
    result = tools.calculate("2 + 2 * 3")
    assert result == "8.0"


def test_calculate_invalid_chars(tools: RetailFarfanTools):
    result = tools.calculate("import os")
    assert "Error" in result


def test_transfer_to_human_agents(tools: RetailFarfanTools):
    result = tools.transfer_to_human_agents("Cliente exige hablar con humano")
    assert "Transfer successful" in result


# ============================================================
# Verificaciones booleanas (env_assertions)
# ============================================================

def test_get_order_status(tools: RetailFarfanTools):
    assert tools.get_order_status("ORD1") == "pending"
    assert tools.get_order_status("ORD2") == "delivered"


def test_order_status_equals(tools: RetailFarfanTools):
    assert tools.order_status_equals("ORD1", "pending") is True
    assert tools.order_status_equals("ORD1", "cancelled") is False


def test_order_contains_and_excludes_product(tools: RetailFarfanTools):
    assert tools.order_contains_product("ORD1", "P1") is True
    assert tools.order_excludes_product("ORD1", "P2") is True
    assert tools.order_excludes_product("ORD1", "P1") is False


def test_get_order_product_ids(tools: RetailFarfanTools):
    result = tools.get_order_product_ids("ORD1")
    ids = json.loads(result)
    assert ids == ["P1"]


# ============================================================
# DB reset
# ============================================================

def test_db_reset_clears_session_state(tools: RetailFarfanTools):
    tools.send_verification_sms("U1")
    tools.verify_sms_code("U1", "1234")
    assert tools.db.users["U1"].verified is True

    tools.db.reset()

    assert tools.db.users["U1"].verified is False
    assert tools.db.users["U1"].current_sms_code is None


def test_db_get_statistics(tools: RetailFarfanTools):
    stats = tools.db.get_statistics()
    assert stats["num_products"] == 4
    assert stats["num_users"] == 5
    assert stats["num_orders"] == 5