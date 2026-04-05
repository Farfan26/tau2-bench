from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
from tau2.domains.enosa_masias.utils import ENOSA_DB_PATH
from tau2.environment.db import DB

# Estados y tipos específicos de ENOSA
TicketStatus = Literal["open", "in_progress", "resolved", "escalated_to_emergency"]
SupplyStatus = Literal["active", "disconnected_due_to_debt"]
IssueType = Literal["power_outage", "public_hazard", "billing", "street_lighting"]

class User(BaseModel):
    user_id: str = Field(description="DNI del cliente")
    full_name: str = Field(description="Nombre completo del titular")
    email: Optional[str] = Field(default="", description="Correo electrónico")
    phone: Optional[str] = Field(default="", description="Teléfono de contacto")

class Supply(BaseModel):
    supply_id: str = Field(description="Identificador interno del suministro")
    supply_number: str = Field(description="Código de suministro (S-XXX)")
    owner_id: str = Field(description="DNI del titular")
    address: str = Field(description="Dirección del predio")
    status: SupplyStatus = Field(description="Estado actual del servicio")
    debt_amount: float = Field(description="Monto total de deuda pendiente")

class Ticket(BaseModel):
    ticket_id: str = Field(description="Código único del ticket de atención")
    user_id: str = Field(description="DNI de quien reporta la incidencia")
    reporter_name: str = Field(description="Nombre registrado en el reporte")
    supply_number: Optional[str] = Field(default=None, description="Número de suministro afectado")
    issue_type: str = Field(description="Tipo de falla o consulta")
    description: str = Field(description="Detalle del problema")
    status: TicketStatus = Field(description="Estado del ticket")
    creation_date: str = Field(description="Fecha de registro (YYYY-MM-DD)")

class EnosaInfo(BaseModel):
    company_name: str = Field(description="Nombre de la empresa")
    city: str = Field(description="Sede principal")
    emergency_phone: str = Field(description="Teléfono de emergencias 24h")
    office_hours: str = Field(description="Horario de atención presencial")

class EnosaDB(DB):
    enosa_info: EnosaInfo = Field(description="Información corporativa")
    users: Dict[str, User] = Field(description="Base de datos de clientes")
    supplies: Dict[str, Supply] = Field(description="Base de datos de suministros")
    tickets: Dict[str, Ticket] = Field(description="Registro de tickets y reclamos")

    def get_statistics(self) -> dict[str, Any]:
        return {
            "empresa": self.enosa_info.company_name,
            "total_clientes": len(self.users),
            "total_suministros": len(self.supplies),
            "total_tickets": len(self.tickets),
        }

def get_db() -> EnosaDB:
    return EnosaDB.load(ENOSA_DB_PATH)