import json
from .data_model import DivemotorDB


def generar_db():
    base_users = {
        "c1": {"id": "c1", "nombre": "Empresa Logistica SAC", "tipo": "empresa", "presupuesto": 80000},
        "c2": {"id": "c2", "nombre": "Juan Perez", "tipo": "persona", "presupuesto": 20000},
        "c3": {"id": "c3", "nombre": "Maria Lopez", "tipo": "persona", "presupuesto": 30000},
        "c4": {"id": "c4", "nombre": "Transporte Norte", "tipo": "empresa", "presupuesto": 100000},
        "c5": {"id": "c5", "nombre": "Luis Garcia", "tipo": "persona", "presupuesto": 15000}
    }

    db = DivemotorDB(
        clientes=base_users,  #  modelo sigue funcionando
        vehiculos={
            "v1": {"id": "v1", "nombre": "Camion Volvo FH", "tipo": "camion", "precio": 50000, "stock": 3},
            "v2": {"id": "v2", "nombre": "Bus Mercedes", "tipo": "bus", "precio": 70000, "stock": 2},
            "v3": {"id": "v3", "nombre": "Auto Hyundai", "tipo": "auto", "precio": 15000, "stock": 0}
        },
        cotizaciones={},
        pedidos={}
    )

    #  aquí agregamos users manualmente para el bot
    data = db.model_dump()
    data["users"] = base_users

    with open("data/tau2/domains/divemotor_santiago/db.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    generar_db()