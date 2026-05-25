import json
import os
import hashlib
import random
from typing import List
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from .data_model import Order, Return, Payment


class RetailTools(ToolKitBase):

    def __init__(self, db):
        super().__init__(db)
        self.db = db

    # =========================
    # NUEVO: MÉTODO REQUERIDO POR EL EVALUADOR
    # =========================
    def get_db_hash(self):
        """
        Retorna el hash de la base de datos actual para que el sistema
        pueda verificar cambios.
        """
        # Convertimos la BD completa a un diccionario y luego a string para hashear
        db_data = {
            "users": {uid: u.__dict__ for uid, u in self.db.users.items()},
            "products": {pid: p.__dict__ for pid, p in self.db.products.items()},
            "orders": {oid: o.__dict__ for oid, o in self.db.orders.items()},
            "returns": {rid: r.__dict__ for rid, r in self.db.returns.items()},
            "payments": {pid: p.__dict__ for pid, p in self.db.payments.items()},
        }
        db_string = json.dumps(db_data, sort_keys=True)
        return hashlib.sha256(db_string.encode()).hexdigest()

    # =========================
    # SEGURIDAD Y SMS
    # =========================
    @is_tool(ToolType.WRITE)
    def send_sms_code(self, user_id: str):
        """Envía un código SMS al teléfono del usuario para verificación de identidad."""
        if user_id not in self.db.users:
            raise Exception(f"Usuario {user_id} no existe")

        codigo = str(random.randint(1000, 9999))
        if not hasattr(self.db, "sms_codes"):
            self.db.sms_codes = {}
        self.db.sms_codes[user_id] = codigo

        return f"Código SMS enviado con éxito. (Nota interna del sistema: El código generado es {codigo}. Pídeselo al usuario y verifica que coincida antes de continuar)."

    # =========================
    # USUARIO
    # =========================
    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str):
        """Obtiene la información detallada de un usuario por su ID."""
        if user_id not in self.db.users:
            raise Exception(f"Usuario {user_id} no existe")
        return self.db.users[user_id]

    # =========================
    # PRODUCTOS
    # =========================
    @is_tool(ToolType.READ)
    def search_products(self, keyword: str):
        """Busca productos por nombre, palabra clave o ID exacto (product_id)."""
        return [
            p
            for p in self.db.products.values()
            if keyword.lower() in p.nombre.lower() or keyword.upper() == p.product_id
        ]

    # =========================
    # PEDIDOS
    # =========================
    @is_tool(ToolType.WRITE)
    def create_order(self, user_id: str, product_ids: List[str]):
        """Crea un nuevo pedido validando stock, estado del producto y del usuario."""
        user = self.get_user_details(user_id)
        if user.estado != "activo":
            raise Exception("Usuario bloqueado")

        total = 0
        for pid in product_ids:
            product = self.db.products.get(pid)
            if not product:
                raise Exception(f"Producto {pid} no existe")
            if product.estado == "descontinuado":
                raise Exception(f"Producto {pid} está descontinuado")
            if product.stock <= 0:
                raise Exception(f"Producto {pid} sin stock")

            total += product.precio
            product.stock -= 1

        order_id = f"ORD{len(self.db.orders) + 1}"
        order = Order(
            order_id=order_id,
            user_id=user_id,
            productos=product_ids,
            total=total,
            estado="pendiente",
        )
        self.db.orders[order_id] = order
        return order

    @is_tool(ToolType.WRITE)
    def cancel_order(self, order_id: str):
        """Cancela un pedido solo si está en estado pendiente o enviado."""
        order = self.db.orders.get(order_id)
        if not order:
            raise Exception("Pedido no existe")
        if order.estado not in ["pendiente", "enviado"]:
            raise Exception(f"No se puede cancelar un pedido en estado: {order.estado}")

        order.estado = "cancelado"
        return order

    @is_tool(ToolType.READ)
    def track_order(self, order_id: str):
        """Consulta el estado actual de un pedido."""
        if order_id not in self.db.orders:
            raise Exception("Pedido no existe")
        return self.db.orders[order_id]

    # =========================
    # DEVOLUCIONES
    # =========================
    @is_tool(ToolType.WRITE)
    def request_return(self, order_id: str, reason: str):
        """Procesa una devolución si el pedido fue entregado y el producto lo permite."""
        order = self.track_order(order_id)
        if order.estado != "entregado":
            raise Exception("El pedido debe estar entregado para solicitar devolución")

        for pid in order.productos:
            product = self.db.products.get(pid)
            if product and not product.permite_devolucion:
                raise Exception(f"El producto {product.nombre} no acepta devoluciones")

        for r in self.db.returns.values():
            if r.order_id == order_id:
                raise Exception("Ya existe una devolución para este pedido")

        return_id = f"RET{len(self.db.returns) + 1}"
        new_return = Return(
            return_id=return_id, order_id=order_id, motivo=reason, estado="solicitada"
        )
        self.db.returns[return_id] = new_return
        return new_return

    # =========================
    # PAGOS
    # =========================
    @is_tool(ToolType.WRITE)
    def process_payment(self, order_id: str, method: str):
        """Registra el pago de un pedido evitando duplicidad."""
        if order_id not in self.db.orders:
            raise Exception("Pedido no existe")

        for p in self.db.payments.values():
            if p.order_id == order_id:
                raise Exception("Este pedido ya fue pagado")

        payment_id = f"PAY{len(self.db.payments) + 1}"
        payment = Payment(
            payment_id=payment_id,
            order_id=order_id,
            metodo_pago=method,
            estado="pagado",
        )
        self.db.payments[payment_id] = payment
        return payment
