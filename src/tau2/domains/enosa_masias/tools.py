from typing import Optional
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.enosa_masias.data_model import EnosaDB

class EnosaToolKit(ToolKitBase):
    """Agent tools for ENOSA domain."""

    db: EnosaDB

    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> dict:
        """Retrieves user details by DNI."""
        user = self.db.users.get(user_id)
        if not user:
            return {"error": f"User with ID {user_id} not found."}
        return user.model_dump()

    @is_tool(ToolType.READ)
    def get_supply_details(self, supply_number: str) -> dict:
        """Retrieves electricity supply details and debt."""
        supply = self.db.supplies.get(supply_number)
        if not supply:
            return {"error": f"Supply {supply_number} not found."}
        return supply.model_dump()

    @is_tool(ToolType.READ)
    def get_ticket_status(self, ticket_id: str) -> dict:
        """Retrieves the current status of an existing ticket."""
        ticket = self.db.tickets.get(ticket_id)
        if not ticket:
            return {"error": f"Ticket {ticket_id} not found."}
        return ticket.model_dump()

    @is_tool(ToolType.WRITE)
    def create_ticket(
        self,
        reporter_id: str,
        issue_type: str,
        description: str,
        supply_number: Optional[str] = None
    ) -> dict:
        """Creates a new support or emergency ticket."""
        user = self.db.users.get(reporter_id)
        if not user:
            return {"error": f"User {reporter_id} not found."}

        valid_types = ["power_outage", "public_hazard", "billing", "street_lighting"]
        if issue_type not in valid_types:
            return {"error": f"Invalid issue type."}

        new_id = f"T{str(len(self.db.tickets) + 1).zfill(3)}"
        status = "escalated_to_emergency" if issue_type == "public_hazard" else "open"

        from tau2.domains.enosa_masias.data_model import Ticket
        new_ticket = Ticket(
            ticket_id=new_id,
            reporter_id=reporter_id,
            supply_number=supply_number,
            issue_type=issue_type,
            description=description,
            status=status,
            creation_date="2026-04-04"
        )
        self.db.tickets[new_id] = new_ticket
        return {"message": "Ticket created.", "ticket": new_ticket.model_dump()}

    @is_tool(ToolType.READ)
    def get_office_locations(self, district: str) -> dict:
        """Returns the address and business hours of ENOSA physical offices by district."""
        offices = {
            "piura": "Sede Central: Av. Sanchez Cerro 1230. Hours: Mon-Fri 08:00 - 17:00",
            "castilla": "Agencia Castilla: Av. Progreso 450. Hours: Mon-Fri 08:00 - 16:00",
            "sullana": "Agencia Sullana: Calle San Martin 500. Hours: Mon-Fri 08:00 - 17:00"
        }
        dist = district.lower()
        if dist in offices:
            return {"district": district, "address_and_hours": offices[dist]}
        return {"error": f"No ENOSA office found in district: {district}. Please direct them to Piura."}