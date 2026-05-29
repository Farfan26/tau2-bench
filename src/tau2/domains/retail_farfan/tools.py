import random
import string
from typing import Annotated

# 1. Importa ToolType desde el paquete del framework
from tau2.environment.toolkit import ToolKitBase, is_tool, ToolType

from tau2.domains.retail_farfan.data_model import RetailDB, Order, Payment, Return


class RetailTools(ToolKitBase):
    """
    Herramientas del agente para el dominio retail_farfan.
    Permite consultar usuarios, productos, pedidos, procesar pagos y devoluciones.
    """

    def __init__(self, db: RetailDB):
        self.db = db

    # ------------------------------------------------------------------
    # 1. get_user_details (READ)
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def get_user_details(
        self,
        user_id: Annotated[str, "ID del usuario a consultar (ej. 'U1')"],
    ) -> dict:
        user = self.db.users.get(user_id)
        if user is None:
            return {"error": f"Usuario '{user_id}' no encontrado."}
        return user.model_dump()

    # ------------------------------------------------------------------
    # 2. search_products (READ)
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def search_products(
        self,
        keyword: Annotated[str, "Palabra clave para buscar productos"],
    ) -> list[dict]:
        keyword_lower = keyword.lower()
        results = [
            p.model_dump()
            for p in self.db.products.values()
            if keyword_lower in p.nombre.lower() or keyword_lower in p.categoria.lower()
        ]
        if not results:
            return [{"message": f"No se encontraron productos para '{keyword}'."}]
        return results

    # ------------------------------------------------------------------
    # 3. create_order (WRITE)
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def create_order(
        self,
        user_id: Annotated[str, "ID del usuario"],
        product_ids: Annotated[list[str], "IDs de productos"],
    ) -> dict:
        user = self.db.users.get(user_id)
        if user is None or user.estado != "activo":
            return {"error": "Usuario inválido o bloqueado."}

        errores, productos_validos, total = [], [], 0.0
        for pid in product_ids:
            product = self.db.products.get(pid)
            if not product or product.estado != "activo" or product.stock <= 0:
                errores.append(f"Producto '{pid}' no disponible.")
                continue
            productos_validos.append(product)
            total += product.precio

        if errores:
            return {"error": " | ".join(errores)}

        order_id = f"ORD{len(self.db.orders) + 1}"
        for product in productos_validos:
            product.stock -= 1

        new_order = Order(
            order_id=order_id,
            user_id=user_id,
            productos=product_ids,
            total=round(total, 2),
            estado="pendiente",
        )
        self.db.orders[order_id] = new_order
        return {"success": True, "order_id": order_id, "message": "Pedido creado."}

    # ------------------------------------------------------------------
    # 4. cancel_order (WRITE)
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def cancel_order(self, order_id: Annotated[str, "ID pedido"]) -> dict:
        order = self.db.orders.get(order_id)
        if not order or order.estado in ["entregado", "cancelado"]:
            return {"error": "Pedido no encontrado o no cancelable."}
        order.estado = "cancelado"
        return {"success": True, "estado": "cancelado"}

    # ------------------------------------------------------------------
    # 5. track_order (READ)
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def track_order(self, order_id: Annotated[str, "ID pedido"]) -> dict:
        order = self.db.orders.get(order_id)
        if not order:
            return {"error": "No encontrado."}
        return {"order_id": order.order_id, "estado": order.estado}

    # ------------------------------------------------------------------
    # 6. request_return (WRITE)
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def request_return(
        self, order_id: Annotated[str, "ID pedido"], reason: Annotated[str, "Motivo"]
    ) -> dict:
        order = self.db.orders.get(order_id)
        if not order or order.estado != "entregado":
            return {"error": "Solo pedidos entregados."}

        return_id = f"RET{len(self.db.returns) + 1}"
        self.db.returns[return_id] = Return(
            return_id=return_id, order_id=order_id, motivo=reason, estado="solicitada"
        )
        return {"success": True, "return_id": return_id}

    # ------------------------------------------------------------------
    # 7. send_sms_code (WRITE)
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def send_sms_code(self, user_id: Annotated[str, "ID usuario"]) -> dict:
        user = self.db.users.get(user_id)
        if not user:
            return {"error": "No encontrado."}
        code = "".join(random.choices(string.digits, k=4))
        self.db.sms_codes[user_id] = code
        return {"success": True, "code": code}

    # ------------------------------------------------------------------
    # 8. process_payment (WRITE)
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def process_payment(
        self,
        order_id: Annotated[str, "ID pedido"],
        method: Annotated[str, "Método"],
        sms_code: Annotated[str, "Código"],
    ) -> dict:
        order = self.db.orders.get(order_id)
        if not order or sms_code != self.db.sms_codes.get(order.user_id):
            return {"error": "Validación fallida."}

        payment_id = f"PAY{len(self.db.payments) + 1}"
        self.db.payments[payment_id] = Payment(
            payment_id=payment_id,
            order_id=order_id,
            metodo_pago=method,
            estado="pagado",
        )
        return {"success": True, "payment_id": payment_id}

    # ------------------------------------------------------------------
    # 9. transfer_to_human (GENERIC)
    # ------------------------------------------------------------------
    @is_tool(ToolType.GENERIC)
    def transfer_to_human(self) -> dict:
        return {"success": True, "message": "Escalado a humano."}
