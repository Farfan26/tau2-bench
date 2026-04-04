from typing import Optional
from tau2.environment.toolkit import ToolKitBase
from tau2.domains.enosa_masias.data_model import EnosaDB

class EnosaToolKit(ToolKitBase):
    """Agent tools for ENOSA domain."""

    db: EnosaDB

    def get_user_details(self, user_id: str) -> dict:
        """
        Retrieves user details.
        Use this to verify identity before processing requests.
        """
        user = self.db.users.get(user_id)
        if not user:
            return {"error": f"User with ID {user_id} not found."}
        return user.model_dump()

    def get_supply_details(self, supply_number: str) -> dict:
        """
        Retrieves electricity supply details, including debt and status.
        """
        supply = self.db.supplies.get(supply_number)
        if not supply:
            return {"error": f"Supply {supply_number} not found."}
        return supply.model_dump()

    def get_ticket_status(self, ticket_id: str) -> dict:
        """
        Retrieves the current status of an existing support ticket.
        """
        ticket = self.db.tickets.get(ticket_id)
        if not ticket:
            return {"error": f"Ticket {ticket_id} not found."}
        return ticket.model_dump()

    def create_ticket(
        self,
        reporter_id: str,
        issue_type: str,
        description: str,
        supply_number: Optional[str] = None
    ) -> dict:
        """
        Creates a new support or emergency ticket.
        Valid types: power_outage, public_hazard, billing, street_lighting.
        """
        user = self.db.users.get(reporter_id)
        if not user:
            return {"error": f"User {reporter_id} not found. Cannot create ticket."}

        valid_types = ["power_outage", "public_hazard", "billing", "street_lighting"]
        if issue_type not in valid_types:
            return {"error": f"Invalid issue type. Must be one of: {valid_types}"}

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
        
        msg = f"Ticket {new_id} created successfully."
        if issue_type == "public_hazard":
            msg += " EMERGENCY ACTIVATED. Stay away from the hazard."
            
        return {
            "message": msg,
            "ticket": new_ticket.model_dump()
        }