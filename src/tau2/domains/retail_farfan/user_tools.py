from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.retail_farfan.data_model import RetailFarfanDB

class RetailFarfanUserTools(ToolKitBase):
    """Herramientas del lado del usuario simulado para interactuar con el entorno."""
    
    db: RetailFarfanDB

    def __init__(self, db: RetailFarfanDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def read_sms_code(self, user_id: str) -> str:
        """
        Simula que el usuario revisa su teléfono móvil para obtener el código 
        de seguridad enviado por el agente.
        
        Args:
            user_id: ID del usuario que intenta validar su identidad.
        """
        if user_id in self.db.users:
            code = self.db.users[user_id].current_sms_code
            if code:
                return f"El código recibido en el SMS es: {code}"
            return "No se ha recibido ningún código SMS aún."
        return "Usuario no encontrado."