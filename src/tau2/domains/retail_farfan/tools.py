"""Toolkit for the retail_farfan domain."""

import json
import random
from typing import List

from tau2.domains.retail_farfan.data_model import (
    GiftCard,
    Order,
    OrderPayment,
    PaymentMethod,
    Product,
    RetailFarfanDB,
    User,
    UserAddress,
    Variant,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class RetailFarfanTools(ToolKitBase):
    """All the tools for the retail_farfan domain."""

    db: RetailFarfanDB

    def __init__(self, db: RetailFarfanDB) -> None:
        super().__init__(db)

    # --- Private Helper Methods (Lógica Interna) ---

    def _get_order(self, order_id: str) -> Order:
        if order_id not in self.db.orders:
            raise ValueError("Order not found")
        return self.db.orders[order_id]

    def _get_user(self, user_id: str) -> User:
        if user_id not in self.db.users:
            raise ValueError("User not found")
        return self.db.users[user_id]

    def _get_product(self, product_id: str) -> Product:
        if product_id not in self.db.products:
            raise ValueError("Product not found")
        return self.db.products[product_id]

    def _get_variant(self, product_id: str, variant_id: str) -> Variant:
        product = self._get_product(product_id)
        if variant_id not in product.variants:
            raise ValueError("Variant not found")
        return product.variants[variant_id]

    def _get_payment_method(self, user_id: str, payment_method_id: str) -> PaymentMethod:
        user = self._get_user(user_id)
        if payment_method_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        return user.payment_methods[payment_method_id]

    def _is_pending_order(self, order: Order) -> bool:
        return "pending" in order.status

    # --- Public Tools (Invocadas por el Agente Gemma) ---

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.READ)
    def get_customer_profile(self, customer_id: str) -> User:
        """Get the profile details of a customer/user, including their verification logs, email, and order history.
        Use this tool as a first diagnostic step to verify if a user account is active or blocked.

        Args:
            customer_id: The unique identifier for the customer, such as 'U1' or 'U3'.
        """
        return self._get_user(customer_id)

    @is_tool(ToolType.READ)
    def get_order_details(self, order_id: str) -> Order:
        """Get the current fulfillment status, item contents, and transaction history of an order.

        Args:
            order_id: The unique identifier for the order, such as 'ORD1' or 'ORD2'.
        """
        return self._get_order(order_id)

    @is_tool(ToolType.READ)
    def search_products(self, query: str) -> str:
        """Search the store inventory for products matching a keyword. Returns names and prices.

        Args:
            query: The text keyword to search for, such as 'laptop'.
        """
        results = {}
        for p_id, product in self.db.products.items():
            if query.lower() in product.name.lower():
                results[product.name] = {
                    "product_id": product.product_id,
                    "variants": {v_id: {"price": v.price, "available": v.available} for v_id, v in product.variants.items()}
                }
        return json.dumps(results, sort_keys=True)

    @is_tool(ToolType.WRITE)
    def send_verification_sms(self, customer_id: str) -> str:
        """Generate and send a secure 4-digit verification code to the customer's registered phone number.
        This is a mandatory first step of the two-factor identity validation cascade.

        Args:
            customer_id: The target customer id to receive the SMS.
        """
        user = self._get_user(customer_id)
        # Código fijo para simulación determinista o aleatorio controlado
        code = "1234" 
        user.current_sms_code = code
        return f"Verification code sent via SMS to customer {customer_id}."

    @is_tool(ToolType.WRITE)
    def verify_sms_code(self, customer_id: str, code: str) -> str:
        """Verify the 4-digit code provided by the customer. Changes their verification status to True if correct.
        Must be checked prior to executing high-risk financial write modifications.

        Args:
            customer_id: The customer id attempting verification.
            code: The 4-digit code provided verbally by the user.
        """
        user = self._get_user(customer_id)
        if user.current_sms_code and user.current_sms_code == code:
            user.verified = True
            return "Success: Identity verification CONFIRMED."
        raise ValueError("Error: Invalid verification code. Identity match failed.")

    @is_tool(ToolType.WRITE)
    def process_refund(self, order_id: str, reason: str) -> Order:
        """Process a cancellation and monetary refund for a pending or valid order.
        Requires prior SMS identity verification via verify_sms_code.

        Args:
            order_id: The target order id to refund, such as 'ORD1'.
            reason: Must be either 'no longer needed' or 'ordered by mistake'.
        """
        order = self._get_order(order_id)
        user = self._get_user(order.user_id)

        # 🔒 Control de Seguridad Rúbrica Entrega 2 y 3
        if not user.verified:
            raise ValueError("Operation denied: Customer identity is unverified. Perform SMS challenge first.")

        if order.status != "pending":
            raise ValueError("Non-pending or completed orders cannot be directly cancelled/refunded via this endpoint")

        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("Invalid reason provided")

        refunds = []
        for payment in order.payment_history:
            refund = OrderPayment(
                transaction_type="refund",
                amount=payment.amount,
                payment_method_id=payment.payment_method_id,
            )
            refunds.append(refund)
            
            # Reembolso inmediato si fue tarjeta de regalo
            payment_method = self._get_payment_method(user.user_id, payment.payment_method_id)
            if isinstance(payment_method, GiftCard):
                payment_method.balance = round(payment_method.balance + payment.amount, 2)

        order.status = "cancelled"
        order.cancel_reason = reason
        order.payment_history.extend(refunds)
        return order

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the user conversation to a human supervisor. 
        Only use this if the user persistently demands a person, or the task falls entirely out of scope.

        Args:
            summary: A concise description of why the escalation is triggered.
        """
        return "Transfer successful to human team."


if __name__ == "__main__":
    from tau2.domains.retail_farfan.data_model import RETAIL_FARFAN_DB_PATH
    try:
        tools = RetailFarfanTools(RetailFarfanDB.load(RETAIL_FARFAN_DB_PATH))
        print("Tools cargadas exitosamente para retail_farfan.")
    except Exception as e:
        print(f"Aviso de inicialización: {e}")
