# src/tau2/domains/retail_farfan/data_model.py
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from tau2.environment.db import DB


class User(BaseModel):
    user_id: str = Field(description="Identificador único del usuario")
    nombre: str = Field(description="Nombre completo del usuario")
    email: str = Field(description="Correo electrónico del usuario")
    telefono: str = Field(description="Número de teléfono de contacto")
    direccion: str = Field(description="Dirección de envío del usuario")
    estado: Literal["activo", "bloqueado"] = Field(
        description="Estado de la cuenta. Solo puede estar activo o bloqueado."
    )


class Product(BaseModel):
    product_id: str = Field(description="Identificador único del producto")
    nombre: str = Field(description="Nombre comercial del producto")
    categoria: str = Field(description="Categoría a la que pertenece el producto")
    precio: float = Field(description="Precio del producto en moneda local")
    stock: int = Field(description="Cantidad de unidades disponibles en inventario")
    estado: Literal["disponible", "descontinuado"] = Field(
        description="Estado comercial del producto"
    )
    permite_devolucion: bool = Field(
        description="Indica si las políticas permiten devolver este producto"
    )


class Order(BaseModel):
    order_id: str = Field(description="Identificador único del pedido")
    user_id: str = Field(description="ID del usuario que realizó la compra")
    productos: List[str] = Field(description="Lista de IDs de los productos comprados")
    total: float = Field(description="Monto total a pagar por el pedido")
    estado: Literal["pendiente", "enviado", "entregado", "cancelado"] = Field(
        description="Estado actual del pedido"
    )


class Return(BaseModel):
    return_id: str = Field(description="Identificador único de la devolución")
    order_id: str = Field(description="ID del pedido asociado a la devolución")
    motivo: str = Field(description="Razón por la cual se devuelve el producto")
    estado: Literal["solicitada", "aprobada", "rechazada"] = Field(
        description="Estado del proceso de devolución"
    )


class Payment(BaseModel):
    payment_id: str = Field(description="Identificador único del pago")
    order_id: str = Field(description="ID del pedido que se está pagando")
    metodo_pago: Literal["credit_card", "paypal", "transferencia"] = Field(
        description="Método utilizado para realizar el pago"
    )
    estado: Literal["pagado", "pendiente", "fallido"] = Field(
        description="Estado de la transacción financiera"
    )


class RetailDB(DB):
    users: Dict[str, User] = Field(
        description="Diccionario de usuarios indexado por user_id"
    )
    products: Dict[str, Product] = Field(
        description="Diccionario de productos indexado por product_id"
    )
    orders: Dict[str, Order] = Field(
        description="Diccionario de pedidos indexado por order_id"
    )
    returns: Dict[str, Return] = Field(
        description="Diccionario de devoluciones indexado por return_id"
    )
    payments: Dict[str, Payment] = Field(
        description="Diccionario de pagos indexado por payment_id"
    )
