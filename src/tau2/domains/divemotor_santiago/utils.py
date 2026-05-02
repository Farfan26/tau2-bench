import json
from .data_model import DivemotorDB


def generar_db():
    db = DivemotorDB(
        clientes={
            "c1": {"id": "c1", "nombre": "Empresa Logistica SAC", "tipo": "empresa", "presupuesto": 80000},
            "c2": {"id": "c2", "nombre": "Juan Perez", "tipo": "persona", "presupuesto": 20000}
        },
        vehiculos={
            "v1": {"id": "v1", "nombre": "Camion Volvo FH", "tipo": "camion", "precio": 50000, "stock": 3},
            "v2": {"id": "v2", "nombre": "Bus Mercedes", "tipo": "bus", "precio": 70000, "stock": 2},
            "v3": {"id": "v3", "nombre": "Auto Hyundai", "tipo": "auto", "precio": 15000, "stock": 0}
        },
        cotizaciones={},
        pedidos={}
    )

    db.dump("data/tau2/domains/divemotor_santiago/db.json")


if __name__ == "__main__":
    generar_db()