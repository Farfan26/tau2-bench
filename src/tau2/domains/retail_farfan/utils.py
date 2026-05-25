# src/tau2/domains/retail_farfan/utils.py
import os
import json
from pathlib import Path

from tau2.domains.retail_farfan.data_model import (
    User,
    Product,
    Order,
    Return,
    Payment,
    RetailDB,
)
from tau2.domains.retail_farfan.tools import RetailTools

# ==========================================
# RUTAS DE DATOS (Data Paths)
# ==========================================
# En tau2-bench, los datos van en la carpeta 'data', no en 'src'
BASE_DATA_PATH = Path("data/tau2/domains/retail_farfan")
RETAIL_DB_PATH = str(BASE_DATA_PATH / "db.json")
RETAIL_POLICY_PATH = str(BASE_DATA_PATH / "policy.md")
RETAIL_TASK_SET_PATH = str(BASE_DATA_PATH / "tasks.json")


# ==========================================
# FUNCIÓN DE CARGA
# ==========================================
def load_retail_farfan_domain():
    """
    Función de carga para el dominio retail_farfan.
    Carga la base de datos desde db.json e inicializa las herramientas.
    """
    # Validación: Si el archivo no existe, el programa se detiene avisándote
    if not os.path.exists(RETAIL_DB_PATH):
        raise FileNotFoundError(
            f"ERROR: No se encontró 'db.json' en la ruta: {RETAIL_DB_PATH}. "
            "Por favor, ejecuta 'python -m src.tau2.domains.retail_farfan.utils' para generarlo."
        )

    # Crear la base de datos usando el método de la clase base DB
    db = RetailDB.load(RETAIL_DB_PATH)

    # Inicializar tus herramientas (RetailTools) pasándole la base de datos
    tools = RetailTools(db)

    # Retornar el objeto que registra tu dominio en el sistema
    return {
        "name": "retail_farfan",
        "db": db,
        "tools": tools,
        "policy_file": RETAIL_POLICY_PATH,
        "tasks_file": RETAIL_TASK_SET_PATH,
    }


# ==========================================
# GENERADOR DE BASE DE DATOS PROGRAMÁTICO
# ==========================================
def generate_db():
    """Genera la base de datos programáticamente usando los modelos Pydantic."""
    print("Generando base de datos...")

    users = {
        "U1": User(
            user_id="U1",
            nombre="Dany",
            email="dany@mail.com",
            telefono="999111222",
            direccion="Lima",
            estado="activo",
        ),
        "U2": User(
            user_id="U2",
            nombre="Ana",
            email="ana@mail.com",
            telefono="988777666",
            direccion="Cusco",
            estado="activo",
        ),
        "U3": User(
            user_id="U3",
            nombre="Luis",
            email="luis@mail.com",
            telefono="977666555",
            direccion="Piura",
            estado="bloqueado",
        ),
        "U4": User(
            user_id="U4",
            nombre="Maria",
            email="maria@mail.com",
            telefono="966555444",
            direccion="Arequipa",
            estado="activo",
        ),
        "U5": User(
            user_id="U5",
            nombre="Carlos",
            email="carlos@mail.com",
            telefono="955444333",
            direccion="Trujillo",
            estado="activo",
        ),
    }

    products = {
        "P1": Product(
            product_id="P1",
            nombre="Laptop Gamer",
            categoria="tech",
            precio=3500.0,
            stock=5,
            estado="disponible",
            permite_devolucion=True,
        ),
        "P2": Product(
            product_id="P2",
            nombre="Mouse Logitech",
            categoria="tech",
            precio=80.0,
            stock=10,
            estado="disponible",
            permite_devolucion=True,
        ),
        "P3": Product(
            product_id="P3",
            nombre="Televisor 55",
            categoria="tech",
            precio=2200.0,
            stock=0,
            estado="disponible",
            permite_devolucion=False,
        ),
        "P4": Product(
            product_id="P4",
            nombre="Audifonos Sony",
            categoria="tech",
            precio=300.0,
            stock=8,
            estado="disponible",
            permite_devolucion=True,
        ),
        "P5": Product(
            product_id="P5",
            nombre="Tablet Samsung",
            categoria="tech",
            precio=1200.0,
            stock=3,
            estado="descontinuado",
            permite_devolucion=True,
        ),
    }

    orders = {
        "ORD1": Order(
            order_id="ORD1",
            user_id="U1",
            productos=["P1"],
            total=3500.0,
            estado="pendiente",
        ),
        "ORD2": Order(
            order_id="ORD2",
            user_id="U2",
            productos=["P2"],
            total=80.0,
            estado="entregado",
        ),
        "ORD3": Order(
            order_id="ORD3",
            user_id="U1",
            productos=["P2"],
            total=80.0,
            estado="entregado",
        ),
        "ORD4": Order(
            order_id="ORD4",
            user_id="U4",
            productos=["P1"],
            total=3500.0,
            estado="entregado",
        ),
        "ORD5": Order(
            order_id="ORD5",
            user_id="U5",
            productos=["P4"],
            total=300.0,
            estado="enviado",
        ),
    }

    returns = {
        "RET1": Return(
            return_id="RET1",
            order_id="ORD4",
            motivo="Fuera de tiempo",
            estado="rechazada",
        )
    }

    payments = {
        "PAY1": Payment(
            payment_id="PAY1",
            order_id="ORD1",
            metodo_pago="credit_card",
            estado="pagado",
        )
    }

    # Ensamblar la base de datos
    db = RetailDB(
        users=users,
        products=products,
        orders=orders,
        returns=returns,
        payments=payments,
    )

    # Crear los directorios si no existen y guardar el JSON
    os.makedirs(BASE_DATA_PATH, exist_ok=True)
    db.dump(RETAIL_DB_PATH)
    print(f"¡Éxito! Base de datos validada y guardada en {RETAIL_DB_PATH}")


# Este bloque ejecuta la generación cuando corres el archivo directamente
if __name__ == "__main__":
    generate_db()
