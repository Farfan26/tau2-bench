"""
Script para generar automáticamente data/tau2/domains/retail_farfan/db.json

Genera los datos mínimos necesarios para que las 20 tareas de tasks.json
funcionen correctamente:
  - Usuarios: U1, U2, U3, U4, U5
  - Productos: P1 (Laptop Gamer), P2 (Mouse Logitech), P3 (Televisor), P4 (Teclado Mecanico)
  - Órdenes: ORD1, ORD2, ORD3, ORD4, ORD5
"""

import os

from tau2.domains.retail_farfan.data_model import (
    CreditCard,
    GiftCard,
    Order,
    OrderItem,
    OrderPayment,
    Paypal,
    Product,
    RetailFarfanDB,
    User,
    UserAddress,
    UserName,
    Variant,
)

# Ruta donde se guardará la base de datos
DB_PATH = "data/tau2/domains/retail_farfan/db.json"


def generate_database() -> RetailFarfanDB:
    # ==================================================
    # 1. Productos
    # ==================================================
    laptop_variant = Variant(
        item_id="1000000001",
        options={"color": "negro", "ram": "16GB"},
        available=True,
        price=3500.00,
    )
    laptop_product = Product(
        name="Laptop Gamer",
        product_id="P1",
        variants={"V1A": laptop_variant},
    )

    mouse_variant = Variant(
        item_id="1000000002",
        options={"color": "negro"},
        available=True,
        price=89.90,
    )
    mouse_product = Product(
        name="Mouse Logitech",
        product_id="P2",
        variants={"V2A": mouse_variant},
    )

    tv_variant = Variant(
        item_id="1000000003",
        options={"pulgadas": "55"},
        available=False,  # sin stock, para la Tarea 19
        price=1800.00,
    )
    tv_product = Product(
        name="Televisor",
        product_id="P3",
        variants={"V3A": tv_variant},
    )

    keyboard_variant = Variant(
        item_id="1000000004",
        options={"color": "blanco"},
        available=True,
        price=250.00,
    )
    keyboard_product = Product(
        name="Teclado Mecanico",
        product_id="P4",
        variants={"V4A": keyboard_variant},
    )

    products = {
        "P1": laptop_product,
        "P2": mouse_product,
        "P3": tv_product,
        "P4": keyboard_product,
    }

    # ==================================================
    # 2. Direcciones reutilizables
    # ==================================================
    addr_u1 = UserAddress(
        address1="Av. Bernal 123", address2="", city="Sechura",
        country="Peru", state="Piura", zip="20631",
    )
    addr_u2 = UserAddress(
        address1="Calle Bernal 456", address2="", city="Sechura",
        country="Peru", state="Piura", zip="20631",
    )
    addr_u3 = UserAddress(
        address1="Jr. Bahia 789", address2="", city="Sechura",
        country="Peru", state="Piura", zip="20631",
    )
    addr_u4 = UserAddress(
        address1="Av. Costanera 321", address2="", city="Sechura",
        country="Peru", state="Piura", zip="20631",
    )
    addr_u5 = UserAddress(
        address1="Malecon 654", address2="", city="Sechura",
        country="Peru", state="Piura", zip="20631",
    )

    # ==================================================
    # 3. Usuarios
    # ==================================================
    user_1 = User(
        user_id="U1",
        name=UserName(first_name="Mario", last_name="Farfan"),
        address=addr_u1,
        email="mario.farfan@example.com",
        phone="999111222",
        payment_methods={
            "PM1": CreditCard(source="credit_card", id="PM1", brand="visa", last_four="4242"),
        },
        orders=["ORD1", "ORD3"],
        verified=False,
        current_sms_code=None,
        is_blocked=False,
    )

    user_2 = User(
        user_id="U2",
        name=UserName(first_name="Ana", last_name="Tume"),
        address=addr_u2,
        email="ana.tume@example.com",
        phone="999222333",
        payment_methods={
            "PM2": Paypal(source="paypal", id="PM2"),
        },
        orders=["ORD2"],
        verified=False,
        current_sms_code=None,
        is_blocked=False,
    )

    user_3 = User(
        user_id="U3",
        name=UserName(first_name="Luis", last_name="Bancayan"),
        address=addr_u3,
        email="luis.bancayan@example.com",
        phone="999333444",
        payment_methods={
            "PM3": GiftCard(source="gift_card", id="PM3", balance=50.00),
        },
        orders=[],
        verified=False,
        current_sms_code=None,
        is_blocked=True,  # 🔒 cuenta bloqueada (Tareas 2 y 19)
    )

    user_4 = User(
        user_id="U4",
        name=UserName(first_name="Rosa", last_name="Pingo"),
        address=addr_u4,
        email="rosa.pingo@example.com",
        phone="999444555",
        payment_methods={
            "PM4": CreditCard(source="credit_card", id="PM4", brand="mastercard", last_four="1111"),
        },
        orders=["ORD4"],
        verified=False,
        current_sms_code=None,
        is_blocked=True,  # 🔒 cuenta bloqueada (Tarea 7)
    )

    user_5 = User(
        user_id="U5",
        name=UserName(first_name="Carlos", last_name="Sechura"),
        address=addr_u5,
        email="carlos.sechura@example.com",
        phone="999555666",
        payment_methods={
            "PM5": CreditCard(source="credit_card", id="PM5", brand="visa", last_four="9999"),
        },
        orders=["ORD5"],
        verified=False,
        current_sms_code=None,
        is_blocked=False,
    )

    users = {
        "U1": user_1,
        "U2": user_2,
        "U3": user_3,
        "U4": user_4,
        "U5": user_5,
    }

    # ==================================================
    # 4. Órdenes
    # ==================================================
    order_1 = Order(
        order_id="ORD1",
        user_id="U1",
        address=addr_u1,
        items=[
            OrderItem(
                name="Laptop Gamer", product_id="P1", item_id="1000000001",
                price=3500.00, options={"color": "negro", "ram": "16GB"},
            ),
        ],
        status="pending",  # Tarea 3: cancelable; Tarea 5/12/15/20: consultas
        fulfillments=[],
        payment_history=[
            OrderPayment(transaction_type="payment", amount=3500.00, payment_method_id="PM1"),
        ],
        created_at="2026-05-01",
    )

    order_2 = Order(
        order_id="ORD2",
        user_id="U2",
        address=addr_u2,
        items=[
            OrderItem(
                name="Mouse Logitech", product_id="P2", item_id="1000000002",
                price=89.90, options={"color": "negro"},
            ),
        ],
        status="delivered",  # Tarea 4/13/14: no cancelable
        fulfillments=[],
        payment_history=[
            OrderPayment(transaction_type="payment", amount=89.90, payment_method_id="PM2"),
        ],
        created_at="2026-04-20",
    )

    order_3 = Order(
        order_id="ORD3",
        user_id="U1",
        address=addr_u1,
        items=[
            OrderItem(
                name="Teclado Mecanico", product_id="P4", item_id="1000000004",
                price=250.00, options={"color": "blanco"},
            ),
        ],
        status="delivered",  # Tarea 6 (devolución) / 18 (reembolso cuenta externa)
        fulfillments=[],
        payment_history=[
            OrderPayment(transaction_type="payment", amount=250.00, payment_method_id="PM1"),
        ],
        created_at="2026-05-10",
    )

    order_4 = Order(
        order_id="ORD4",
        user_id="U4",
        address=addr_u4,
        items=[
            OrderItem(
                name="Televisor", product_id="P3", item_id="1000000003",
                price=1800.00, options={"pulgadas": "55"},
            ),
        ],
        status="delivered",  # Tarea 7: usuario bloqueado, pedido entregado
        fulfillments=[],
        payment_history=[
            OrderPayment(transaction_type="payment", amount=1800.00, payment_method_id="PM4"),
        ],
        created_at="2026-04-15",
    )

    order_5 = Order(
        order_id="ORD5",
        user_id="U5",
        address=addr_u5,
        items=[
            OrderItem(
                name="Mouse Logitech", product_id="P2", item_id="1000000002",
                price=89.90, options={"color": "negro"},
            ),
        ],
        status="pending",  # Tareas 8 y 9: pago con/sin verificación SMS correcta
        fulfillments=[],
        payment_history=[],
        created_at="2026-06-01",
    )

    orders = {
        "ORD1": order_1,
        "ORD2": order_2,
        "ORD3": order_3,
        "ORD4": order_4,
        "ORD5": order_5,
    }

    # ==================================================
    # 5. Ensamblar la base de datos
    # ==================================================
    db = RetailFarfanDB(products=products, users=users, orders=orders)
    return db


def main():
    db = generate_database()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        f.write(db.model_dump_json(indent=2))

    stats = db.get_statistics()
    print(f"Base de datos generada exitosamente en: {DB_PATH}")
    print(f"Estadisticas: {stats}")


if __name__ == "__main__":
    main()