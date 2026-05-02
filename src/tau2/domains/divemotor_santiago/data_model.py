from pydantic import BaseModel
from typing import Dict
from tau2.environment.db import DB


class Cliente(BaseModel):
    id: str
    nombre: str
    tipo: str  # empresa / persona
    presupuesto: float


class Vehiculo(BaseModel):
    id: str
    nombre: str
    tipo: str  # camion / bus / auto
    precio: float
    stock: int


class Cotizacion(BaseModel):
    id: str
    cliente_id: str
    vehiculo_id: str
    precio_final: float
    estado: str  # pendiente / aprobada / rechazada


class Pedido(BaseModel):
    id: str
    cotizacion_id: str
    estado: str  # confirmado / cancelado


class DivemotorDB(DB):
    clientes: Dict[str, Cliente]
    vehiculos: Dict[str, Vehiculo]
    cotizaciones: Dict[str, Cotizacion]
    pedidos: Dict[str, Pedido]