from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from tau2.environment.db import DB

# --- Configuración de Ruta Base de Datos de tu Dominio ---
RETAIL_FARFAN_DB_PATH = "data/tau2/domains/retail_farfan/db.json"


# ==========================================
# 1. MODELOS DE PRODUCTOS E INVENTARIO
# ==========================================

class Variant(BaseModel):
    """Represents a specific variant of a product with its options, availability and price"""
    item_id: str = Field(description="Unique identifier for the variant, such as '1008292230'")
    options: Dict[str, str] = Field(
        description="Dictionary of option names to values (e.g. {'color': 'blue', 'size': 'large'})"
    )
    available: bool = Field(description="Whether this variant is currently in stock")
    price: float = Field(description="Price of this variant")


class Product(BaseModel):
    """Represents a product with its variants"""
    name: str = Field(description="Name of the product, such as 'Laptop Gamer' or 'Mouse Logitech'")
    product_id: str = Field(description="Unique identifier for the product, such as 'P1' or 'P2'")
    variants: Dict[str, Variant] = Field(
        description="Dictionary of variants indexed by variant ID"
    )


# ==========================================
# 2. MODELOS DE CLIENTES / USUARIOS Y SEGURIDAD SMS
# ==========================================

class UserName(BaseModel):
    """Represents a user's full name"""
    first_name: str = Field(description="User's first name")
    last_name: str = Field(description="User's last name")


class UserAddress(BaseModel):
    """Represents a physical address"""
    address1: str = Field(description="Primary address line")
    address2: str = Field(description="Secondary address line (can be empty string)")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


class PaymentMethodBase(BaseModel):
    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    source: Literal["credit_card"] = Field(
        description="Indicates this is a credit card payment method"
    )
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")


class Paypal(PaymentMethodBase):
    source: Literal["paypal"] = Field(
        description="Indicates this is a paypal payment method"
    )


class GiftCard(PaymentMethodBase):
    source: Literal["gift_card"] = Field(
        description="Indicates this is a gift card payment method"
    )
    balance: float = Field(description="Gift card value amount available")
    id: str = Field(description="Unique identifier for the gift card")


PaymentMethod = Union[CreditCard, GiftCard, Paypal]


class User(BaseModel):
    """Represents a user with personal information, payment methods, order history, and verification states"""
    user_id: str = Field(description="Unique identifier for the user, such as 'U1' or 'U3'")
    name: UserName = Field(description="User's full name")
    address: UserAddress = Field(description="User's primary default address")
    email: str = Field(description="User's registered email address")
    phone: str = Field(description="User's cell phone number required for SMS challenge logs", default="555-0100")
    payment_methods: Dict[str, PaymentMethod] = Field(
        description="Dictionary of payment methods indexed by payment method ID"
    )
    orders: List[str] = Field(description="List of order IDs associated with this user")

    # 🔒 Atributos Críticos contra Ataques Adversarios (Entrega 2 y 3)
    verified: bool = Field(
        description="Whether the customer identity has been successfully confirmed via SMS code during this active session",
        default=False
    )
    current_sms_code: Optional[str] = Field(
        description="The last 4-digit verification code sent via SMS system challenge to this specific customer",
        default=None
    )
    is_blocked: bool = Field(
        description="Whether the customer account is blocked due to policy violations",
        default=False
    )


# ==========================================
# 3. MODELOS DE ÓRDENES, SEGUIMIENTO Y PAGOS
# ==========================================

class OrderFullfilment(BaseModel):
    """Represents the fulfillment details for items in an order"""
    tracking_id: List[str] = Field(description="List of tracking IDs for shipments")
    item_ids: List[str] = Field(
        description="List of item IDs included in this fulfillment shipment"
    )


class OrderItem(BaseModel):
    """Represents an item inside an order"""
    name: str = Field(description="Name of the product type")
    product_id: str = Field(description="ID of the product parent")
    item_id: str = Field(description="ID of the specific variant purchased")
    price: float = Field(description="Price of the item at the exact time of purchase")
    options: Dict[str, str] = Field(description="Specific options chosen for this item (e.g. color, size)")


OrderPaymentType = Literal["payment", "refund"]


class OrderPayment(BaseModel):
    """Represents a financial payment or refund transaction log for an order"""
    transaction_type: OrderPaymentType = Field(
        description="Type of transaction (payment or refund)"
    )
    amount: float = Field(description="Amount of the transaction")
    payment_method_id: str = Field(description="ID of the payment method used")


OrderStatus = Literal[
    "processed",
    "pending",
    "pending (item modified)",
    "delivered",
    "cancelled",
    "exchange requested",
    "return requested",
]

CancelReason = Literal["no longer needed", "ordered by mistake", "defective product", "other"]


class Order(BaseModel):
    """Represents an active order transaction with all its products, status, shipment logs, and payment histories"""
    order_id: str = Field(description="Unique identifier for the order, such as 'ORD1' or 'ORD2'")
    user_id: str = Field(description="Unique identifier for the buyer user account")
    address: UserAddress = Field(description="Shipping address utilized for this specific order execution")
    items: List[OrderItem] = Field(description="List of items contents in the order allocation")
    status: OrderStatus = Field(description="Current workflow status state of the order assignment")
    fulfillments: List[OrderFullfilment] = Field(
        description="List of tracking shipments and fulfillments of the order lifecycle"
    )
    payment_history: List[OrderPayment] = Field(description="Ledger of payment and refund transactions history logs")
    created_at: str = Field(description="ISO Date when the order entry was registered (YYYY-MM-DD)", default="2026-01-01")
    cancel_reason: Optional[CancelReason] = Field(
        description="Reason for cancelling the order entry. Must be 'no longer needed' or 'ordered by mistake'",
        default=None,
    )
    exchange_items: Optional[List[str]] = Field(
        description="List of item IDs requested to be exchanged", default=None
    )
    exchange_new_items: Optional[List[str]] = Field(
        description="List of new variant item IDs chosen to replace old entries", default=None
    )
    exchange_payment_method_id: Optional[str] = Field(
        description="Payment method ID processing price adjustments for the exchange", default=None
    )
    exchange_price_difference: Optional[float] = Field(
        description="Calculated monetary price difference for the asset exchange execution", default=None
    )
    return_items: Optional[List[str]] = Field(
        description="List of item IDs requested to be returned", default=None
    )
    return_payment_method_id: Optional[str] = Field(
        description="Payment method ID routing the refund calculation for returns", default=None
    )
    return_id: Optional[str] = Field(
        description="Identifier assigned to a registered return request", default=None
    )


# ==========================================
# 4. BASE DE DATOS DEL DOMINIO RETAIL FARFAN
# ==========================================

class RetailFarfanDB(DB):
    """Database engine context mapping all items, customer accounts, and order logs for retail_farfan"""
    products: Dict[str, Product] = Field(
        description="Dictionary of all product items indexed by their main product ID string"
    )
    users: Dict[str, User] = Field(
        description="Dictionary of all registered user assets indexed by unique user ID string"
    )
    orders: Dict[str, Order] = Field(
        description="Dictionary of all active purchases indexed by order ID string"
    )

    def reset(self) -> None:
        """
        Mandatory framework hook. Cleans session authentication tokens and dynamic
        SMS states between different evaluation simulation steps to prevent state-bleeding.
        """
        for user in self.users.values():
            user.verified = False
            user.current_sms_code = None

    def get_statistics(self) -> dict[str, Any]:
        """Calculates dynamic ledger size attributes for diagnostic assertions metrics."""
        num_products = len(self.products)
        num_users = len(self.users)
        num_orders = len(self.orders)
        total_num_items = sum(
            len(product.variants) for product in self.products.values()
        )
        return {
            "num_products": num_products,
            "num_users": num_users,
            "num_orders": num_orders,
            "total_num_items": total_num_items,
        }


def get_db():
    """Loads up the static JSON resource data model into the structural DB instance wrapper class."""
    return RetailFarfanDB.load(RETAIL_FARFAN_DB_PATH)


if __name__ == "__main__":
    try:
        db = get_db()
        print("--- [RETAIL FARFAN] BASE DE DATOS COMPILADA CORRECTAMENTE ---")
        print(f"Estadísticas cargadas: {db.get_statistics()}")
    except Exception as e:
        print(f"[Aviso] Estructura del data_model.py validada correctamente.")
        print(f"Nota: db.json no cargado aún en la ruta estática: {e}")