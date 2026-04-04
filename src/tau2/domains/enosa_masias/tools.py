from typing import Optional
from tau2.domains.enosa_masias.data_model import EnosaDB, User, Supply, Ticket
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

class EnosaTools(ToolKitBase):
    """Tools for the ENOSA customer service domain."""

    db: EnosaDB

    def __init__(self, db: EnosaDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def get_user(self, user_id: str) -> User:
        """
        Get the user details by their ID (DNI).

        Args:
            user_id: The ID (DNI) of the user.

        Returns:
            The user details.

        Raises:
            ValueError: If the user is not found.
        """
        if user_id not in self.db.users:
            raise ValueError(f"User with ID '{user_id}' not found in the system.")
        return self.db.users[user_id]

    @is_tool(ToolType.READ)
    def get_supply(self, supply_number: str) -> Supply:
        """
        Get the electricity supply details by its supply number.

        Args:
            supply_number: The supply number (e.g., 'S-1001').

        Returns:
            The supply details including status and owner ID.

        Raises:
            ValueError: If the supply is not found.
        """
        if supply_number not in self.db.supplies:
            raise ValueError(f"Supply '{supply_number}' not found.")
        return self.db.supplies[supply_number]

    @is_tool(ToolType.WRITE)
    def create_ticket(
        self,
        reporter_id: str,
        issue_type: str,
        description: str,
        supply_number: Optional[str] = None
    ) -> Ticket:
        """
        Create a support or emergency ticket for ENOSA.

        Args:
            reporter_id: The DNI of the user reporting the issue.
            issue_type: The type of issue ('power_outage', 'public_hazard', 'street_lighting', 'billing').
            description: A detailed description of the issue.
            supply_number: The supply number affected (optional, can be left empty for public hazards).

        Returns:
            The created ticket.

        Raises:
            ValueError: If the issue_type is invalid.
        """
        valid_issues = ["power_outage", "public_hazard", "street_lighting", "billing"]
        if issue_type not in valid_issues:
            raise ValueError(f"Invalid issue_type. Must be one of: {valid_issues}")

        ticket_id = f"TKT-{len(self.db.tickets) + 1:04d}"
        
        # If it's a public hazard, it automatically escalates to emergency
        status = "escalated_to_emergency" if issue_type == "public_hazard" else "open"

        ticket = Ticket(
            ticket_id=ticket_id,
            reporter_id=reporter_id,
            supply_number=supply_number,
            issue_type=issue_type,
            description=description,
            status=status,
        )
        self.db.tickets[ticket_id] = ticket
        return ticket

    def assert_ticket_exists(self, reporter_id: str, issue_type: str) -> bool:
        """
        Check whether a ticket exists for a given reporter and issue type.
        """
        for ticket in self.db.tickets.values():
            if ticket.reporter_id == reporter_id and ticket.issue_type == issue_type:
                return True
        return False