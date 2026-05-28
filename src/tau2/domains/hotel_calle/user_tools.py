from tau2.domains.hotel_calle.user_data_model import HotelCalleUserDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HotelCalleUserTools(ToolKitBase):
    """Tools that simulate actions available to the hotel guest."""

    db: HotelCalleUserDB

    def __init__(self, db: HotelCalleUserDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def read_latest_sms_code(self, reservation_id: str) -> str:
        """
        Read the latest SMS verification code received for a reservation.

        Args:
            reservation_id: Reservation identifier associated with the SMS.
        """
        for message in reversed(self.db.sms_messages):
            if message.reservation_id == reservation_id:
                return message.code
        return "No SMS verification code has been received for this reservation."

    @is_tool(ToolType.READ)
    def read_latest_sms_message(self, reservation_id: str) -> dict:
        """
        Read the latest SMS verification message received for a reservation.

        Args:
            reservation_id: Reservation identifier associated with the SMS.
        """
        for message in reversed(self.db.sms_messages):
            if message.reservation_id == reservation_id:
                return message.model_dump()
        return {"reservation_id": reservation_id, "received": False}
