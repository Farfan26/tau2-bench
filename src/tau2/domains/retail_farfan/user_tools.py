import json
import hashlib


class RetailUserTools:
    """
    Herramientas del lado del usuario (User Simulator).
    Permiten al usuario simulado interactuar con elementos fuera del chat, como su teléfono.
    """

    def __init__(self, db):
        self.db = db

    def get_db_hash(self):
        """
        Calcula el hash de la base de datos para la evaluación.
        Permite al evaluador verificar si hubo cambios tras las acciones.
        """
        # Serializamos los datos relevantes de la base de datos
        db_data = {
            "users": {uid: u.__dict__ for uid, u in self.db.users.items()},
            "products": {pid: p.__dict__ for pid, p in self.db.products.items()},
            "orders": {oid: o.__dict__ for oid, o in self.db.orders.items()},
        }
        # Convertimos a string ordenado para que el hash sea consistente
        db_string = json.dumps(db_data, sort_keys=True)
        return hashlib.sha256(db_string.encode()).hexdigest()

    def check_sms_messages(self, user_id: str) -> str:
        """
        Revisa la bandeja de entrada del teléfono para buscar códigos SMS recientes.
        """
        # Verificamos si el agente guardó un código en la base de datos simulada
        if hasattr(self.db, "sms_codes") and user_id in self.db.sms_codes:
            codigo = self.db.sms_codes[user_id]
            return f"Tienes un nuevo mensaje SMS: 'Tu código de verificación de Retail Farfan es {codigo}. No lo compartas con nadie.'"

        return "Bandeja de entrada vacía. No tienes mensajes SMS nuevos."
