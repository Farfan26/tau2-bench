from pydantic import BaseModel  # type: ignore
from typing import List, Dict

from tau2.environment.db import DB


class User(BaseModel):
    user_id: str
    nombre: str
    email: str
    telefono: str
    direccion: str
    estado: str  # activo | bloqueado


class Product(BaseModel):
    product_id: str
    nombre: str
    categoria: str
    precio: float
    stock: int
    estado: str  # activo | descontinuado
    permite_devolucion: bool


class Order(BaseModel):
    order_id: str
    user_id: str
    productos: List[str]
    total: float
    estado: str  # pendiente | enviado | entregado | cancelado


class Return(BaseModel):
    return_id: str
    order_id: str
    motivo: str
    estado: str  # solicitada | aprobada | rechazada


class Payment(BaseModel):
    payment_id: str
    order_id: str
    metodo_pago: str
    estado: str  # pagado | fallido


class RetailDB(DB):
    users: Dict[str, User] = {}
    products: Dict[str, Product] = {}
    orders: Dict[str, Order] = {}
    returns: Dict[str, Return] = {}
    payments: Dict[str, Payment] = {}
    sms_codes: Dict[str, str] = {}
