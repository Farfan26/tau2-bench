from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB
from tau2.domains.enosa_masias.utils import ENOSA_DB_PATH

# Status Literals to restrict AI choices
SupplyStatus = Literal["active", "disconnected_due_to_debt"]
TicketIssueType = Literal["power_outage", "public_hazard", "street_lighting", "billing"]
TicketStatus = Literal["open", "inspecting", "resolved", "escalated_to_emergency"]

class User(BaseModel):
    user_id: str = Field(description="Unique DNI/ID of the user")
    name: str = Field(description="Full name of the user")
    phone: str = Field(description="Contact phone number")
    email: str = Field(description="Contact email address")

class Supply(BaseModel):
    supply_number: str = Field(description="Unique electricity supply number (e.g., S-1001)")
    owner_id: str = Field(description="DNI of the user who owns this supply")
    address: str = Field(description="Physical address of the supply")
    status: SupplyStatus = Field(description="Current connection status of the electricity supply")

class Ticket(BaseModel):
    ticket_id: str = Field(description="Unique identifier for the support or emergency ticket")
    reporter_id: str = Field(description="DNI of the user who reported the issue")
    supply_number: Optional[str] = Field(description="Supply number affected by the issue, if applicable", default=None)
    issue_type: TicketIssueType = Field(description="Category of the reported issue")
    description: str = Field(description="Detailed description of the customer's problem or emergency")
    status: TicketStatus = Field(description="Current status of the ticket")

class EnosaDB(DB):
    users: Dict[str, User] = Field(
        description="Registered ENOSA users indexed by user_id (DNI)",
        default_factory=dict
    )
    supplies: Dict[str, Supply] = Field(
        description="Electricity supplies indexed by supply_number",
        default_factory=dict
    )
    tickets: Dict[str, Ticket] = Field(
        description="Customer support and emergency tickets indexed by ticket_id",
        default_factory=dict
    )

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_users": len(self.users),
            "num_supplies": len(self.supplies),
            "num_tickets": len(self.tickets),
        }

def get_db():
    return EnosaDB.load(ENOSA_DB_PATH)