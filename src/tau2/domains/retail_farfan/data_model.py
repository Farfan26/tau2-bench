import json
from pydantic import BaseModel
from typing import List, Dict


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


class RetailDB(BaseModel):
    users: Dict[str, User]
    products: Dict[str, Product]
    orders: Dict[str, Order]
    returns: Dict[str, Return]
    payments: Dict[str, Payment]
    sms_codes: Dict[str, str] = {}  # Necesario para el flujo de SMS

    @classmethod
    def load(cls, path: str):
        """Carga la base de datos desde un archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convertimos los diccionarios cargados a los objetos Pydantic correspondientes
        return cls(
            users={k: User(**v) for k, v in data.get("users", {}).items()},
            products={k: Product(**v) for k, v in data.get("products", {}).items()},
            orders={k: Order(**v) for k, v in data.get("orders", {}).items()},
            returns={k: Return(**v) for k, v in data.get("returns", {}).items()},
            payments={k: Payment(**v) for k, v in data.get("payments", {}).items()},
            sms_codes=data.get("sms_codes", {}),
        )
