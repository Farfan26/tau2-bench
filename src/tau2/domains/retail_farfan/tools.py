"""Toolkit for the retail_farfan domain."""

import json
import hashlib
from typing import List, Optional

from tau2.domains.retail_farfan.data_model import (
    GiftCard,
    Order,
    OrderItem,
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

    # ============================================================
    # --- Private Helper Methods ---
    # ============================================================

    def _get_order(self, order_id: str) -> Order:
        if order_id not in self.db.orders:
            raise ValueError(f"Order {order_id} not found.")
        return self.db.orders[order_id]

    def _get_user(self, user_id: str) -> User:
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found.")
        return self.db.users[user_id]

    def _get_product(self, product_id: str) -> Product:
        if product_id not in self.db.products:
            raise ValueError(f"Product {product_id} not found.")
        return self.db.products[product_id]

    def _get_payment_method(self, user_id: str, payment_method_id: str) -> PaymentMethod:
        user = self._get_user(user_id)
        if payment_method_id not in user.payment_methods:
            raise ValueError("Payment method not found.")
        return user.payment_methods[payment_method_id]

    def _first_available_variant(self, product: Product) -> Optional[Variant]:
        for variant in product.variants.values():
            if variant.available:
                return variant
        return None

    def _deterministic_id(self, prefix: str, *parts: str) -> str:
        """Generate a deterministic, reproducible ID based on input data.

        This avoids non-deterministic IDs (e.g. uuid4) which break the
        evaluator's replay mechanism, since the same tool call with the
        same arguments must always return the same result.
        """
        raw = "|".join(parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8].upper()
        return f"{prefix}{digest}"

    # ============================================================
    # --- Public Tools: Cuenta y Perfil ---
    # ============================================================

    @is_tool(ToolType.READ)
    def check_account_status(self, customer_id: str) -> str:
        """Checks if a customer account is blocked or active.
        Use this tool FIRST to diagnose why a purchase or refund might be failing."""
        try:
            user = self._get_user(customer_id)
            status = "BLOCKED" if user.is_blocked else "ACTIVE"
            return f"Customer {customer_id} account status: {status}."
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.READ)
    def get_customer_profile(self, customer_id: str) -> str:
        """Get profile details, email, and order history for a customer."""
        try:
            user = self._get_user(customer_id)
            return json.dumps({
                "user_id": user.user_id,
                "name": f"{user.name.first_name} {user.name.last_name}",
                "verified": user.verified,
                "email": user.email,
                "phone": user.phone,
                "is_blocked": user.is_blocked,
                "orders": user.orders,
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    # ============================================================
    # --- Public Tools: Productos ---
    # ============================================================

    @is_tool(ToolType.READ)
    def search_products(self, query: str) -> str:
        """Search for products by name. Returns matching products with
        their available variants, options, and prices."""
        try:
            query_lower = query.strip().lower()
            results = []
            for product in self.db.products.values():
                if query_lower in product.name.lower():
                    variants_info = []
                    for variant_id, variant in product.variants.items():
                        variants_info.append({
                            "item_id": variant.item_id,
                            "options": variant.options,
                            "available": variant.available,
                            "price": variant.price,
                        })
                    results.append({
                        "product_id": product.product_id,
                        "name": product.name,
                        "variants": variants_info,
                    })
            if not results:
                return f"No products found matching '{query}'."
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.READ)
    def get_product_details(self, product_id: str) -> str:
        """Get full details (variants, prices, availability) of a single product by its product_id."""
        try:
            product = self._get_product(product_id)
            return json.dumps({
                "product_id": product.product_id,
                "name": product.name,
                "variants": {
                    vid: {
                        "item_id": v.item_id,
                        "options": v.options,
                        "available": v.available,
                        "price": v.price,
                    }
                    for vid, v in product.variants.items()
                },
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    # ============================================================
    # --- Public Tools: Órdenes ---
    # ============================================================

    @is_tool(ToolType.READ)
    def get_order_details(self, order_id: str) -> str:
        """Get the full details of an order, including status, items, and payment history."""
        try:
            order = self._get_order(order_id)
            return order.model_dump_json(indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.WRITE)
    def create_order(self, customer_id: str, product_id: str, payment_method_id: Optional[str] = None) -> str:
        """Create a new order for a customer with a single product.
        FAILS if the customer account is blocked or the product has no available variants.
        If payment_method_id is not provided, the order is created as 'pending' (unpaid)."""
        try:
            user = self._get_user(customer_id)

            if user.is_blocked:
                return f"FAIL: Account {customer_id} is blocked. Cannot create order."

            product = self._get_product(product_id)
            variant = self._first_available_variant(product)
            if variant is None:
                return f"FAIL: Product {product_id} ('{product.name}') has no available stock."

            order_id = self._deterministic_id("ORD", customer_id, product_id, str(len(self.db.orders)))

            payment_history: List[OrderPayment] = []
            status = "pending"
            if payment_method_id:
                self._get_payment_method(customer_id, payment_method_id)
                payment_history.append(
                    OrderPayment(
                        transaction_type="payment",
                        amount=variant.price,
                        payment_method_id=payment_method_id,
                    )
                )
                status = "processed"

            new_order = Order(
                order_id=order_id,
                user_id=customer_id,
                address=user.address,
                items=[
                    OrderItem(
                        name=product.name,
                        product_id=product.product_id,
                        item_id=variant.item_id,
                        price=variant.price,
                        options=variant.options,
                    )
                ],
                status=status,
                fulfillments=[],
                payment_history=payment_history,
            )

            self.db.orders[order_id] = new_order
            user.orders.append(order_id)

            return (
                f"Success: Order {order_id} created for customer {customer_id}. "
                f"Total: {variant.price}. Status: {status}."
            )
        except Exception as e:
            return f"FAIL: {str(e)}"

    @is_tool(ToolType.WRITE)
    def cancel_order(self, order_id: str, reason: str) -> str:
        """Cancel an order. Only orders with status 'pending' or
        'pending (item modified)' can be cancelled. 'delivered' orders cannot be cancelled."""
        try:
            order = self._get_order(order_id)

            if order.status == "delivered":
                return f"FAIL: Order {order_id} is already 'delivered' and cannot be cancelled."
            if order.status == "cancelled":
                return f"FAIL: Order {order_id} is already cancelled."
            if order.status not in ("pending", "pending (item modified)", "processed"):
                return f"FAIL: Order {order_id} has status '{order.status}' and cannot be cancelled."

            valid_reasons = ("no longer needed", "ordered by mistake", "defective product", "other")
            if reason not in valid_reasons:
                reason = "other"

            order.status = "cancelled"
            order.cancel_reason = reason
            return f"Success: Order {order_id} has been cancelled. New status: 'cancelled'."
        except Exception as e:
            return f"FAIL: {str(e)}"

    @is_tool(ToolType.WRITE)
    def update_order_items(self, order_id: str, product_id: str) -> str:
        """Replace the items of a pending order with a single new product.
        Only orders with status 'pending' can be modified."""
        try:
            order = self._get_order(order_id)

            if order.status not in ("pending", "pending (item modified)"):
                return f"FAIL: Order {order_id} has status '{order.status}' and items cannot be modified."

            product = self._get_product(product_id)
            variant = self._first_available_variant(product)
            if variant is None:
                return f"FAIL: Product {product_id} ('{product.name}') has no available stock."

            order.items = [
                OrderItem(
                    name=product.name,
                    product_id=product.product_id,
                    item_id=variant.item_id,
                    price=variant.price,
                    options=variant.options,
                )
            ]
            order.status = "pending (item modified)"
            return f"Success: Order {order_id} updated. It now contains only product {product_id} ('{product.name}')."
        except Exception as e:
            return f"FAIL: {str(e)}"

    # ============================================================
    # --- Public Tools: Devoluciones y Reembolsos ---
    # ============================================================

    @is_tool(ToolType.WRITE)
    def request_return(self, order_id: str, reason: str) -> str:
        """Register a return request for an order. The order's status changes to 'cancelled'
        and a return_id is generated. The customer account must not be blocked."""
        try:
            order = self._get_order(order_id)
            user = self._get_user(order.user_id)

            if user.is_blocked:
                return f"FAIL: Account {user.user_id} is blocked. Cannot process return."

            if order.status not in ("delivered", "processed", "pending"):
                return f"FAIL: Order {order_id} has status '{order.status}' and is not eligible for return."

            return_id = self._deterministic_id("RET", order_id, reason)
            order.return_id = return_id
            order.return_items = [item.item_id for item in order.items]
            order.status = "cancelled"

            valid_reasons = ("no longer needed", "ordered by mistake", "defective product", "other")
            order.cancel_reason = reason if reason in valid_reasons else "defective product"

            return (
                f"Success: Return registered for order {order_id}. "
                f"Return ID: {return_id}. New order status: 'cancelled'."
            )
        except Exception as e:
            return f"FAIL: {str(e)}"

    @is_tool(ToolType.WRITE)
    def send_verification_sms(self, customer_id: str) -> str:
        """Sends a 4-digit verification code to the customer's phone.
        REQUIRED before process_refund or paying for an order with a credit card."""
        try:
            user = self._get_user(customer_id)
            if user.is_blocked:
                return "FAIL: Account is blocked. Verification code cannot be sent."
            user.current_sms_code = "1234"
            return f"Verification code sent to {customer_id}."
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.WRITE)
    def verify_sms_code(self, customer_id: str, code: str) -> str:
        """Verify the 4-digit SMS code provided by the customer.
        REQUIRED before process_refund or paying for an order with a credit card."""
        try:
            user = self._get_user(customer_id)
            if user.current_sms_code == code:
                user.verified = True
                return "Success: Identity verified."
            return "Error: Invalid code."
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.WRITE)
    def process_refund(self, order_id: str, reason: str) -> str:
        """Process refund for order_id. ONLY call if identity is verified.
        Only 'pending' orders can be refunded."""
        try:
            order = self._get_order(order_id)
            user = self._get_user(order.user_id)

            if user.is_blocked:
                return "FAIL: Account is blocked. Action denied."

            if not user.verified:
                return "FAIL: Identity not verified. Please call send_verification_sms and verify_sms_code first."

            if order.status != "pending":
                return f"FAIL: Order status is '{order.status}', only 'pending' orders can be refunded."

            for payment in order.payment_history:
                payment_method = self._get_payment_method(user.user_id, payment.payment_method_id)
                if isinstance(payment_method, GiftCard):
                    payment_method.balance = round(payment_method.balance + payment.amount, 2)

            order.status = "cancelled"
            valid_reasons = ("no longer needed", "ordered by mistake", "defective product", "other")
            order.cancel_reason = reason if reason in valid_reasons else "other"
            return f"Success: Refund processed for {order_id}."
        except Exception as e:
            return f"FAIL: {str(e)}"

    # ============================================================
    # --- Public Tools: Pagos ---
    # ============================================================

    @is_tool(ToolType.WRITE)
    def pay_order(self, order_id: str, payment_method_id: str, sms_code: str) -> str:
        """Pay for a pending order. REQUIRES that send_verification_sms was called first
        for the order's owner, and that sms_code matches the code sent (1234).
        On success, order status becomes 'paid'."""
        try:
            order = self._get_order(order_id)
            user = self._get_user(order.user_id)

            if user.is_blocked:
                return f"FAIL: Account {user.user_id} is blocked. Cannot process payment."

            if order.status not in ("pending", "pending (item modified)"):
                return f"FAIL: Order {order_id} has status '{order.status}' and cannot be paid."

            if user.current_sms_code is None:
                return "FAIL: No verification code was sent. Call send_verification_sms first."

            if sms_code != user.current_sms_code:
                return "FAIL: Incorrect verification code. Payment denied."

            payment_method = self._get_payment_method(user.user_id, payment_method_id)
            total = sum(item.price for item in order.items)

            order.payment_history.append(
                OrderPayment(
                    transaction_type="payment",
                    amount=total,
                    payment_method_id=payment_method_id,
                )
            )
            order.status = "paid"
            user.verified = True
            return f"Success: Order {order_id} paid. Amount: {total}. New status: 'paid'."
        except Exception as e:
            return f"FAIL: {str(e)}"

    # ============================================================
    # --- Public Tools: Generales ---
    # ============================================================

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """Calculate the result of a mathematical expression containing only
        numbers and the operators + - * / ( )."""
        if not all(char in "0123456789+-*/(). " for char in expression):
            return "Error: Invalid characters in expression."
        try:
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        except Exception as e:
            return f"Error calculating: {str(e)}"

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """Escalate the conversation to a human supervisor. Use this when the
        customer explicitly and persistently requests a human agent."""
        return "Transfer successful to human team."

    # ============================================================
    # --- Public Tools: Verificaciones de Estado (para ENV_ASSERTION) ---
    # ============================================================

    @is_tool(ToolType.READ)
    def get_order_status(self, order_id: str) -> str:
        """Return the current status string of an order (e.g. 'pending', 'cancelled', 'paid', 'delivered')."""
        try:
            order = self._get_order(order_id)
            return order.status
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.READ)
    def get_order_product_ids(self, order_id: str) -> str:
        """Return the list of product_ids currently contained in an order, as a JSON array."""
        try:
            order = self._get_order(order_id)
            return json.dumps([item.product_id for item in order.items])
        except Exception as e:
            return f"Error: {str(e)}"

    @is_tool(ToolType.READ)
    def order_status_equals(self, order_id: str, status: str) -> bool:
        """Check whether the given order's current status equals the provided status string."""
        try:
            order = self._get_order(order_id)
            return order.status == status
        except Exception:
            return False

    @is_tool(ToolType.READ)
    def order_contains_product(self, order_id: str, product_id: str) -> bool:
        """Check whether the given order currently contains the given product_id."""
        try:
            order = self._get_order(order_id)
            return any(item.product_id == product_id for item in order.items)
        except Exception:
            return False

    @is_tool(ToolType.READ)
    def order_excludes_product(self, order_id: str, product_id: str) -> bool:
        """Check whether the given order does NOT contain the given product_id."""
        try:
            order = self._get_order(order_id)
            return all(item.product_id != product_id for item in order.items)
        except Exception:
            return False