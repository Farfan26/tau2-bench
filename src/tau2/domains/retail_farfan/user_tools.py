class RetailUserTools:
    """
    Herramientas del lado del usuario (User Simulator).
    Permiten al usuario simulado interactuar con elementos fuera del chat, como su teléfono.
    """

    def __init__(self, db):
        self.db = db

    def check_sms_messages(self, user_id: str) -> str:
        """
        Revisa la bandeja de entrada del teléfono para buscar códigos SMS recientes.
        """
        # Verificamos si el agente guardó un código en la base de datos simulada
        if hasattr(self.db, "sms_codes") and user_id in self.db.sms_codes:
            codigo = self.db.sms_codes[user_id]
            return f"Tienes un nuevo mensaje SMS: 'Tu código de verificación de Retail Farfan es {codigo}. No lo compartas con nadie.'"

        return "Bandeja de entrada vacía. No tienes mensajes SMS nuevos."
