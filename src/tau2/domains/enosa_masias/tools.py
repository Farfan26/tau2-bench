from typing import Optional, List
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.enosa_masias.data_model import EnosaDB, Ticket, User, Supply

class EnosaToolKit(ToolKitBase):
    """Herramientas del asistente de atención al cliente de ENOSA."""

    db: EnosaDB

    def __init__(self, db: EnosaDB) -> None:
        super().__init__(db)

    # --- MÉTODOS PRIVADOS ---
    def _get_user(self, user_id: str) -> User:
        if user_id not in self.db.users:
            raise ValueError(f"Cliente con DNI '{user_id}' no registrado en ENOSA")
        return self.db.users[user_id]

    def _get_supply(self, supply_number: str) -> Supply:
        for s in self.db.supplies.values():
            if s.supply_number == supply_number:
                return s
        raise ValueError(f"Suministro '{supply_number}' no encontrado")

    # --- HERRAMIENTAS PÚBLICAS ---
    @is_tool(ToolType.READ)
    def get_enosa_info(self) -> dict:
        """Obtiene información general de ENOSA y contactos de emergencia."""
        return self.db.enosa_info.model_dump()

    @is_tool(ToolType.READ)
    def get_supply_details(self, supply_number: str) -> dict:
        """Consulta deuda, dirección y estado de un suministro específico."""
        return self._get_supply(supply_number).model_dump()

    @is_tool(ToolType.READ)
    def get_ticket(self, ticket_id: str) -> dict:
        """Consulta el estado y detalles de un ticket de atención existente."""
        if ticket_id not in self.db.tickets:
            raise ValueError(f"Ticket '{ticket_id}' no existe en el sistema")
        return self.db.tickets[ticket_id].model_dump()

    @is_tool(ToolType.WRITE)
    def create_ticket(
        self,
        reporter_id: str,
        issue_type: str,
        description: str,
        supply_number: Optional[str] = None
    ) -> Ticket:
        """Registra un nuevo ticket por falla eléctrica, peligro o facturación."""
        user = self._get_user(reporter_id)
        
        valid_types = ["power_outage", "public_hazard", "billing", "street_lighting"]
        if issue_type not in valid_types:
            raise ValueError(f"Tipo de incidencia inválido. Use uno de: {valid_types}")

        t_id = f"T-{len(self.db.tickets) + 1:03d}"
        ticket = Ticket(
            ticket_id=t_id,
            user_id=reporter_id,
            reporter_name=user.full_name,
            supply_number=supply_number,
            issue_type=issue_type,
            description=description.strip(),
            status="escalated_to_emergency" if issue_type == "public_hazard" else "open",
            creation_date="2026-04-05"
        )
        self.db.tickets[t_id] = ticket
        return ticket

    @is_tool(ToolType.READ)
    def search_supplies_by_dni(self, user_id: str) -> list[dict]:
        """Lista todos los suministros eléctricos vinculados a un DNI."""
        self._get_user(user_id)
        return [s.model_dump() for s in self.db.supplies.values() if s.owner_id == user_id]