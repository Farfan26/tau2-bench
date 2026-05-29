"""Toolkit para el dominio retail_farfan."""

import random
import string
from typing import List

from tau2.domains.retail_farfan.data_model import Order, Payment, Return, RetailDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class RetailTools(ToolKitBase):
    """Herramientas del agente para el dominio retail_farfan."""

    db: RetailDB

    def __init__(self, db: RetailDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------
    def _get_user(self, user_id: str):
        if user_id not in self.db.users:
            raise ValueError(f"Usuario '{user_id}' no encontrado.")
        return self.db.users[user_id]

    def _get_order(self, order_id: str):
        if order_id not in self.db.orders:
            raise ValueError(f"Pedido '{order_id}' no encontrado.")
        return self.db.orders[order_id]

    def _get_product(self, product_id: str):
        if product_id not in self.db.products:
            raise ValueError(f"Producto '{product_id}' no encontrado.")
        return self.db.products[product_id]

    def _new_order_id(self) -> str:
        existing = set(self.db.orders.keys())
        for i in range(1, 999):
            oid = f"ORD{i}"
            if oid not in existing:
                return oid
        raise ValueError("No se pueden generar más IDs de pedido.")

    def _new_return_id(self) -> str:
        existing = set(self.db.returns.keys())
        for i in range(1, 999):
            rid = f"RET{i}"
            if rid not in existing:
                return rid
        raise ValueError("No se pueden generar más IDs de devolución.")

    def _new_payment_id(self) -> str:
        existing = set(self.db.payments.keys())
        for i in range(1, 999):
            pid = f"PAY{i}"
            if pid not in existing:
                return pid
        raise ValueError("No se pueden generar más IDs de pago.")

    # ------------------------------------------------------------------
    # 1. get_user_details
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> dict:
        """
        Retorna los detalles de un usuario dado su user_id.
        Usar antes de cualquier acción que involucre al usuario.

        Args:
            user_id: ID del usuario a consultar (ej. 'U1').

        Returns:
            Diccionario con los datos del usuario.

        Raises:
            ValueError: Si el usuario no existe.
        """
        return self._get_user(user_id).model_dump()

    # ------------------------------------------------------------------
    # 2. search_products
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def search_products(self, keyword: str) -> List[dict]:
        """
        Busca productos en el catálogo cuyo nombre o categoría contenga la palabra clave.

        Args:
            keyword: Palabra clave para buscar productos (ej. 'mouse', 'laptop').

        Returns:
            Lista de productos que coinciden con la búsqueda.

        Raises:
            ValueError: Si no se encuentran productos.
        """
        kw = keyword.lower()
        results = [
            p.model_dump()
            for p in self.db.products.values()
            if kw in p.nombre.lower() or kw in p.categoria.lower()
        ]
        if not results:
            raise ValueError(f"No se encontraron productos para '{keyword}'.")
        return results

    # ------------------------------------------------------------------
    # 3. create_order
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def create_order(self, user_id: str, product_ids: List[str]) -> dict:
        """
        Crea un nuevo pedido para el usuario con los productos indicados.
        Valida que el usuario esté activo y que todos los productos tengan stock.
        Reduce el stock de cada producto al crear el pedido.

        Args:
            user_id: ID del usuario que realiza la compra (ej. 'U1').
            product_ids: Lista de IDs de productos a comprar (ej. ['P1', 'P2']).

        Returns:
            Diccionario con los datos del pedido creado.

        Raises:
            ValueError: Si el usuario no existe, está bloqueado, o algún producto no está disponible.
        """
        user = self._get_user(user_id)
        if user.estado != "activo":
            raise ValueError(
                f"El usuario '{user_id}' está bloqueado y no puede realizar compras."
            )

        productos_validos = []
        total = 0.0
        for pid in product_ids:
            product = self._get_product(pid)
            if product.estado != "activo":
                raise ValueError(
                    f"El producto '{pid}' ({product.nombre}) está descontinuado."
                )
            if product.stock <= 0:
                raise ValueError(
                    f"El producto '{pid}' ({product.nombre}) no tiene stock disponible."
                )
            productos_validos.append(product)
            total += product.precio

        order_id = self._new_order_id()

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

        return {
            "order_id": order_id,
            "user_id": user_id,
            "productos": product_ids,
            "total": round(total, 2),
            "estado": "pendiente",
            "message": f"Pedido '{order_id}' creado exitosamente por S/. {round(total, 2)}.",
        }

    # ------------------------------------------------------------------
    # 4. cancel_order
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def cancel_order(self, order_id: str) -> dict:
        """
        Cancela un pedido existente.
        Solo se permite cancelar pedidos en estado 'pendiente' o 'enviado'.

        Args:
            order_id: ID del pedido a cancelar (ej. 'ORD1').

        Returns:
            Diccionario con la confirmación de la cancelación.

        Raises:
            ValueError: Si el pedido no existe, ya fue entregado o ya está cancelado.
        """
        order = self._get_order(order_id)

        if order.estado == "entregado":
            raise ValueError(
                f"El pedido '{order_id}' ya fue entregado y no puede cancelarse."
            )
        if order.estado == "cancelado":
            raise ValueError(f"El pedido '{order_id}' ya está cancelado.")

        order.estado = "cancelado"
        return {
            "order_id": order_id,
            "estado": "cancelado",
            "message": f"Pedido '{order_id}' cancelado exitosamente.",
        }

    # ------------------------------------------------------------------
    # 5. track_order
    # ------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def track_order(self, order_id: str) -> dict:
        """
        Retorna el estado actual de un pedido y su información básica.

        Args:
            order_id: ID del pedido a rastrear (ej. 'ORD1').

        Returns:
            Diccionario con el estado y datos del pedido.

        Raises:
            ValueError: Si el pedido no existe.
        """
        order = self._get_order(order_id)
        descripciones = {
            "pendiente": "Tu pedido está registrado y en preparación.",
            "enviado": "Tu pedido está en camino.",
            "entregado": "Tu pedido fue entregado.",
            "cancelado": "Tu pedido fue cancelado.",
        }
        return {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "productos": order.productos,
            "total": order.total,
            "estado": order.estado,
            "descripcion": descripciones.get(order.estado, "Estado desconocido."),
        }

    # ------------------------------------------------------------------
    # 6. request_return
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def request_return(self, order_id: str, reason: str) -> dict:
        """
        Registra una solicitud de devolución para un pedido entregado.
        El pedido debe estar en estado 'entregado', el producto debe permitir devolución
        y no debe existir una devolución previa para ese pedido.

        Args:
            order_id: ID del pedido para el cual se solicita devolución (ej. 'ORD3').
            reason: Motivo de la devolución (ej. 'defective', 'wrong_item', 'changed_mind').

        Returns:
            Diccionario con los datos de la devolución registrada.

        Raises:
            ValueError: Si el pedido no existe, no está entregado, el producto no permite
                        devolución, o ya existe una devolución previa.
        """
        order = self._get_order(order_id)

        if order.estado != "entregado":
            raise ValueError(
                f"Solo se pueden devolver pedidos entregados. Estado actual: '{order.estado}'."
            )

        for pid in order.productos:
            product = self._get_product(pid)
            if not product.permite_devolucion:
                raise ValueError(
                    f"El producto '{pid}' ({product.nombre}) no permite devoluciones."
                )

        for ret in self.db.returns.values():
            if ret.order_id == order_id:
                raise ValueError(
                    f"Ya existe una solicitud de devolución para el pedido '{order_id}' "
                    f"(estado: {ret.estado})."
                )

        return_id = self._new_return_id()
        new_return = Return(
            return_id=return_id,
            order_id=order_id,
            motivo=reason,
            estado="solicitada",
        )
        self.db.returns[return_id] = new_return

        return {
            "return_id": return_id,
            "order_id": order_id,
            "motivo": reason,
            "estado": "solicitada",
            "message": f"Solicitud de devolución '{return_id}' registrada exitosamente.",
        }

    # ------------------------------------------------------------------
    # 7. send_sms_code
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def send_sms_code(self, user_id: str) -> dict:
        """
        Genera y envía un código de verificación SMS de 4 dígitos al teléfono registrado
        del usuario. Debe llamarse antes de process_payment para autenticar al usuario.

        Args:
            user_id: ID del usuario al que se enviará el código SMS (ej. 'U1').

        Returns:
            Diccionario con confirmación del envío y el teléfono destino.

        Raises:
            ValueError: Si el usuario no existe.
        """
        user = self._get_user(user_id)
        code = "".join(random.choices(string.digits, k=4))
        self.db.sms_codes[user_id] = code

        return {
            "user_id": user_id,
            "telefono": user.telefono,
            "code": code,
            "message": (
                f"Código SMS enviado al número {user.telefono}. "
                "El usuario debe ingresarlo para continuar."
            ),
        }

    # ------------------------------------------------------------------
    # 8. process_payment
    # ------------------------------------------------------------------
    @is_tool(ToolType.WRITE)
    def process_payment(self, order_id: str, method: str, sms_code: str) -> dict:
        """
        Procesa el pago de un pedido existente.
        Requiere verificación previa con código SMS enviado por send_sms_code.

        Args:
            order_id: ID del pedido a pagar (ej. 'ORD5').
            method: Método de pago: 'credit_card', 'debit_card' o 'cash'.
            sms_code: Código SMS de 4 dígitos ingresado por el usuario.

        Returns:
            Diccionario con la confirmación del pago.

        Raises:
            ValueError: Si el pedido no existe, ya fue pagado, o el código SMS es incorrecto.
        """
        order = self._get_order(order_id)

        for pay in self.db.payments.values():
            if pay.order_id == order_id and pay.estado == "pagado":
                raise ValueError(f"El pedido '{order_id}' ya tiene un pago registrado.")

        user_id = order.user_id
        stored_code = self.db.sms_codes.get(user_id)

        if stored_code is None:
            raise ValueError(
                "No se ha enviado un código SMS para este usuario. "
                "Usa send_sms_code primero."
            )
        if sms_code != stored_code:
            raise ValueError("Código SMS incorrecto. Pago denegado por seguridad.")

        del self.db.sms_codes[user_id]

        payment_id = self._new_payment_id()
        new_payment = Payment(
            payment_id=payment_id,
            order_id=order_id,
            metodo_pago=method,
            estado="pagado",
        )
        self.db.payments[payment_id] = new_payment

        return {
            "payment_id": payment_id,
            "order_id": order_id,
            "metodo_pago": method,
            "estado": "pagado",
            "message": f"Pago '{payment_id}' procesado exitosamente para el pedido '{order_id}'.",
        }

    # ------------------------------------------------------------------
    # 9. transfer_to_human
    # ------------------------------------------------------------------
    @is_tool(ToolType.GENERIC)
    def transfer_to_human(self, summary: str) -> str:
        """
        Transfiere la conversación a un agente humano de soporte.
        Usar cuando el usuario exige hablar con un humano o el problema no puede resolverse.

        Args:
            summary: Resumen del problema del usuario para el agente humano.

        Returns:
            Mensaje de confirmación de la transferencia.
        """
        return (
            "Tu caso ha sido escalado a un agente humano de RETAIL_FARFAN. "
            "Un representante se pondrá en contacto contigo en los próximos minutos. "
            "Gracias por tu paciencia."
        )
