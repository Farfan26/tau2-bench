from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from tau2.environment.db import DB
from tau2.domains.enosa_masias.utils import ENOSA_DB_PATH

class User(BaseModel):
    """ENOSA Customer"""
    user_id: str = Field(description="Unique ID (DNI)")
    name: str = Field(description="Full name of the owner")
    phone: str = Field(description="Contact phone")
    email: str = Field(description="Email address")

class Supply(BaseModel):
    """Electricity supply linked to a customer"""
    supply_number: str = Field(description="Supply ID (e.g., S-1001)")
    owner_id: str = Field(description="DNI of the owner")
    address: str = Field(description="Installation address")
    status: str = Field(description="Status: active or disconnected_due_to_debt")
    debt_amount: float = Field(description="Current debt in soles")

class Ticket(BaseModel):
    """Support or emergency ticket"""
    ticket_id: str = Field(description="Unique ticket ID")
    reporter_id: str = Field(description="DNI of the reporter")
    supply_number: Optional[str] = Field(default=None, description="Affected supply")
    issue_type: str = Field(description="Type: power_outage, public_hazard, billing, street_lighting")
    description: str = Field(description="Description of the problem")
    status: str = Field(description="Status: open, in_progress, resolved, escalated_to_emergency")
    creation_date: str = Field(description="Creation date (YYYY-MM-DD)")

class EnosaDB(DB):
    """Database for ENOSA domain"""
    users: Dict[str, User] = Field(default={}, description="Users indexed by user_id")
    supplies: Dict[str, Supply] = Field(default={}, description="Supplies indexed by supply_number")
    tickets: Dict[str, Ticket] = Field(default={}, description="Tickets indexed by ticket_id")

    @classmethod
    def load(cls) -> "EnosaDB":
        return cls.model_validate_json(open(ENOSA_DB_PATH).read())