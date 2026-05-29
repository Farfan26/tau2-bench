import random
import string
from typing import Annotated

from tau2.environment.toolkit import ToolKitBase, is_tool

from tau2.domains.retail_farfan.data_model import RetailDB, Order, Payment, Return


class RetailTools(ToolKitBase):
    """
    Herramientas del agente para el dominio retail_farfan.
    Permite consultar usuarios, productos, pedidos, procesar pagos y devoluciones.
    """

    def __init__(self, db: RetailDB):
        self.db = db

    # ------------------------------------------------------------------
    # 1. get_user_details
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def get_user_details(
        self,
        user_id: Annotated[str, "ID del usuario a consultar (ej. 'U1')"],
    ) -> dict:
        """
        Retorna los detalles de un usuario dado su user_id.
        Usar antes de cualquier acción que involucre al usuario.
        """
        user = self.db.users.get(user_id)  # type: ignore
        if user is None:
            return {"error": f"Usuario '{user_id}' no encontrado."}
        return user.model_dump()

    # ------------------------------------------------------------------
    # 2. search_products
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def search_products(
        self,
        keyword: Annotated[
            str, "Palabra clave para buscar productos por nombre o categoría"
        ],
    ) -> list[dict]:
        """
        Busca productos en el catálogo cuyo nombre o categoría contenga la palabra clave.
        Retorna una lista de productos que coinciden.
        """
        keyword_lower = keyword.lower()
        results = [
            p.model_dump()
            for p in self.db.products.values()  # type: ignore
            if keyword_lower in p.nombre.lower() or keyword_lower in p.categoria.lower()
        ]
        if not results:
            return [{"message": f"No se encontraron productos para '{keyword}'."}]
        return results

    # ------------------------------------------------------------------
    # 3. create_order
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def create_order(
        self,
        user_id: Annotated[str, "ID del usuario que realiza la compra"],
        product_ids: Annotated[
            list[str], "Lista de IDs de productos a comprar (ej. ['P1', 'P2'])"
        ],
    ) -> dict:
        """
        Crea un nuevo pedido para el usuario con los productos indicados.
        Valida que el usuario esté activo y que todos los productos tengan stock disponible.
        Reduce el stock de cada producto al crear el pedido.
        """
        # Validar usuario
        user = self.db.users.get(user_id)  # type: ignore
        if user is None:
            return {"error": f"Usuario '{user_id}' no encontrado."}
        if user.estado != "activo":
            return {
                "error": f"El usuario '{user_id}' está bloqueado y no puede realizar compras."
            }

        # Validar productos
        errores = []
        productos_validos = []
        total = 0.0

        for pid in product_ids:
            product = self.db.products.get(pid)  # type: ignore
            if product is None:
                errores.append(f"Producto '{pid}' no encontrado.")
                continue
            if product.estado != "activo":
                errores.append(
                    f"Producto '{pid}' ({product.nombre}) está descontinuado."
                )
                continue
            if product.stock <= 0:
                errores.append(
                    f"Producto '{pid}' ({product.nombre}) sin stock disponible."
                )
                continue
            productos_validos.append(product)
            total += product.precio

        if errores:
            return {"error": " | ".join(errores)}

        # Generar order_id único
        existing_ids = set(self.db.orders.keys())  # type: ignore
        order_id = f"ORD{len(existing_ids) + 1}"
        while order_id in existing_ids:
            order_id = "ORD" + "".join(random.choices(string.digits, k=4))

        # Reducir stock
        for product in productos_validos:
            product.stock -= 1

        # Crear pedido
        new_order = Order(
            order_id=order_id,
            user_id=user_id,
            productos=product_ids,
            total=round(total, 2),
            estado="pendiente",
        )
        self.db.orders[order_id] = new_order  # type: ignore

        return {
            "success": True,
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
    @is_tool  # type: ignore
    def cancel_order(
        self,
        order_id: Annotated[str, "ID del pedido a cancelar (ej. 'ORD1')"],
    ) -> dict:
        """
        Cancela un pedido existente. Solo se permite cancelar pedidos en estado
        'pendiente' o 'enviado'. No se pueden cancelar pedidos entregados o ya cancelados.
        """
        order = self.db.orders.get(order_id)  # type: ignore
        if order is None:
            return {"error": f"Pedido '{order_id}' no encontrado."}

        if order.estado == "entregado":
            return {
                "error": f"El pedido '{order_id}' ya fue entregado y no puede cancelarse.",
                "estado_actual": order.estado,
            }
        if order.estado == "cancelado":
            return {
                "error": f"El pedido '{order_id}' ya está cancelado.",
                "estado_actual": order.estado,
            }

        order.estado = "cancelado"
        return {
            "success": True,
            "order_id": order_id,
            "estado": "cancelado",
            "message": f"Pedido '{order_id}' cancelado exitosamente.",
        }

    # ------------------------------------------------------------------
    # 5. track_order
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def track_order(
        self,
        order_id: Annotated[str, "ID del pedido a rastrear (ej. 'ORD1')"],
    ) -> dict:
        """
        Retorna el estado actual de un pedido y su información básica.
        Usar para responder consultas de seguimiento de envíos.
        """
        order = self.db.orders.get(order_id)  # type: ignore
        if order is None:
            return {"error": f"Pedido '{order_id}' no encontrado."}

        estados_descripcion = {
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
            "descripcion": estados_descripcion.get(order.estado, "Estado desconocido."),
        }

    # ------------------------------------------------------------------
    # 6. request_return
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def request_return(
        self,
        order_id: Annotated[str, "ID del pedido para el cual se solicita devolución"],
        reason: Annotated[
            str,
            "Motivo de la devolución (ej. 'defective', 'wrong_item', 'changed_mind')",
        ],
    ) -> dict:
        """
        Registra una solicitud de devolución para un pedido entregado.
        El pedido debe estar en estado 'entregado', el producto debe permitir devolución
        y no debe existir una devolución previa para ese pedido.
        """
        order = self.db.orders.get(order_id)  # type: ignore
        if order is None:
            return {"error": f"Pedido '{order_id}' no encontrado."}

        if order.estado != "entregado":
            return {
                "error": f"Solo se pueden devolver pedidos entregados. Estado actual: '{order.estado}'."
            }

        # Verificar que los productos permitan devolución
        for pid in order.productos:
            product = self.db.products.get(pid)  # type: ignore
            if product and not product.permite_devolucion:
                return {
                    "error": f"El producto '{pid}' ({product.nombre}) no permite devoluciones."
                }

        # Verificar que no exista una devolución previa
        for ret in self.db.returns.values():  # type: ignore
            if ret.order_id == order_id:
                return {
                    "error": f"Ya existe una solicitud de devolución para el pedido '{order_id}' (estado: {ret.estado})."
                }

        # Crear devolución
        return_id = f"RET{len(self.db.returns) + 1}"  # type: ignore
        new_return = Return(
            return_id=return_id,
            order_id=order_id,
            motivo=reason,
            estado="solicitada",
        )
        self.db.returns[return_id] = new_return  # type: ignore

        return {
            "success": True,
            "return_id": return_id,
            "order_id": order_id,
            "motivo": reason,
            "estado": "solicitada",
            "message": f"Solicitud de devolución '{return_id}' registrada exitosamente.",
        }

    # ------------------------------------------------------------------
    # 7. send_sms_code
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def send_sms_code(
        self,
        user_id: Annotated[
            str, "ID del usuario al que se enviará el código SMS de verificación"
        ],
    ) -> dict:
        """
        Genera y envía un código de verificación SMS de 4 dígitos al teléfono registrado
        del usuario. Debe llamarse antes de process_payment para autenticar al usuario.
        El código generado se almacena internamente para su validación posterior.
        """
        user = self.db.users.get(user_id)  # type: ignore
        if user is None:
            return {"error": f"Usuario '{user_id}' no encontrado."}

        code = "".join(random.choices(string.digits, k=4))
        self.db.sms_codes[user_id] = code  # type: ignore

        return {
            "success": True,
            "user_id": user_id,
            "telefono": user.telefono,
            "code": code,
            "message": f"Código SMS enviado al número {user.telefono}. El usuario debe ingresarlo para continuar.",
        }

    # ------------------------------------------------------------------
    # 8. process_payment
    # ----------------------------------# type: ignore--------------------------------
    @is_tool  # type: ignore
    def process_payment(
        self,
        order_id: Annotated[str, "ID del pedido a pagar"],
        method: Annotated[str, "Método de pago: 'credit_card', 'debit_card' o 'cash'"],
        sms_code: Annotated[
            str, "Código SMS de 4 dígitos ingresado por el usuario para verificación"
        ],
    ) -> dict:
        """
        Procesa el pago de un pedido existente. Requiere verificación previa con código SMS
        enviado por send_sms_code. El pago se rechaza si el código no coincide, si el pedido
        no existe, o si ya fue pagado anteriormente.
        """
        order = self.db.orders.get(order_id)  # type: ignore
        if order is None:
            return {"error": f"Pedido '{order_id}' no encontrado."}

        # Verificar pago duplicado
        for pay in self.db.payments.values():  # type: ignore
            if pay.order_id == order_id and pay.estado == "pagado":
                return {"error": f"El pedido '{order_id}' ya tiene un pago registrado."}

        # Validar código SMS
        user_id = order.user_id
        stored_code = self.db.sms_codes.get(user_id)  # type: ignore

        if stored_code is None:
            return {
                "error": "No se ha enviado un código SMS para este usuario. Usa send_sms_code primero."
            }
        if sms_code != stored_code:
            return {
                "error": "Código SMS incorrecto. Pago denegado por seguridad.",
                "hint": "Solicita un nuevo código con send_sms_code e inténtalo de nuevo.",
            }

        # Limpiar código usado
        del self.db.sms_codes[user_id]  # type: ignore

        # Registrar pago
        payment_id = f"PAY{len(self.db.payments) + 1}"  # type: ignore
        new_payment = Payment(
            payment_id=payment_id,
            order_id=order_id,
            metodo_pago=method,
            estado="pagado",
        )
        self.db.payments[payment_id] = new_payment  # type: ignore

        return {
            "success": True,
            "payment_id": payment_id,
            "order_id": order_id,
            "metodo_pago": method,
            "estado": "pagado",
            "message": f"Pago '{payment_id}' procesado exitosamente para el pedido '{order_id}'.",
        }

    # ------------------------------------------------------------------
    # 9. transfer_to_human
    # ------------------------------------------------------------------
    @is_tool  # type: ignore
    def transfer_to_human(self) -> dict:
        """
        Transfiere la conversación a un agente humano de soporte.
        Usar cuando el usuario exige hablar con un humano, cuando el problema no puede
        resolverse con las herramientas disponibles, o cuando hay escalamiento necesario.
        """
        return {
            "success": True,
            "message": (
                "Tu caso ha sido escalado a un agente humano. "
                "Un representante de RETAIL_FARFAN se pondrá en contacto contigo "
                "en los próximos minutos. Gracias por tu paciencia."
            ),
        }
