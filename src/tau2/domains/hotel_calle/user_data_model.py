from pydantic import Field

from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


class SmsMessage(BaseModelNoExtra):
    reservation_id: str = Field(description="Reservation associated with the SMS")
    phone: str = Field(description="Phone number that received the SMS")
    role: str = Field(description="Role associated with the verification code")
    code: str = Field(description="Verification code received by SMS")


class HotelCalleUserDB(DB):
    sms_messages: list[SmsMessage] = Field(
        default_factory=list,
        description="SMS messages visible to the simulated user",
    )
