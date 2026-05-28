from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.hotel_calle.utils import HOTEL_CALLE_DB_PATH
from tau2.environment.db import DB

ReservationStatus = Literal["confirmed", "cancelled", "pending"]
RoomStatus = Literal["available", "reserved", "maintenance"]


class User(BaseModel):
    user_id: str = Field(description="Unique identifier for a guest profile")
    full_name: str = Field(description="Full name of the guest")
    email: Optional[str] = Field(default="", description="Email address of the guest")
    phone: Optional[str] = Field(default="", description="Phone number of the guest")


class RoomType(BaseModel):
    room_type_id: str = Field(description="Unique identifier for the room type")
    name: str = Field(description="Public name of the room type")
    description: str = Field(description="Short description of the room type")
    bed_configuration: str = Field(description="Description of the beds in the room")
    price_per_night: float = Field(description="Nightly room price")
    available_rooms: int = Field(
        description="How many rooms of this type are currently available"
    )
    max_guests: int = Field(description="Maximum number of guests allowed")
    amenities: list[str] = Field(description="Amenities included with the room")


class HotelRoom(BaseModel):
    room_id: str = Field(description="Unique identifier for a physical hotel room")
    room_number: str = Field(description="Human-readable room number")
    room_type_id: str = Field(description="Identifier of the room type")
    floor: int = Field(description="Floor where the room is located")
    status: RoomStatus = Field(description="Current status of the room")
    notes: str = Field(description="Optional operational notes for the room")


class Reservation(BaseModel):
    reservation_id: str = Field(description="Unique identifier of the reservation")
    user_id: str = Field(description="Identifier of the guest profile")
    guest_name: str = Field(description="Full name of the guest")
    guest_email: Optional[str] = Field(
        default="",
        description="Email address of the guest when it was collected",
    )
    guest_phone: Optional[str] = Field(
        default="",
        description="Phone number of the guest when it was collected",
    )
    room_type_id: str = Field(description="Identifier of the reserved room type")
    room_type_name: str = Field(description="Name of the reserved room type")
    room_id: Optional[str] = Field(default=None, description="Reserved physical room identifier")
    room_number: Optional[str] = Field(default=None, description="Reserved physical room number")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(description="Number of guests included in the reservation")
    total_price: float = Field(description="Total reservation price")
    special_request: str = Field(description="Optional special request from the guest")
    status: ReservationStatus = Field(description="Current reservation status")


class VerificationCode(BaseModel):
    reservation_id: str = Field(description="Reservation being verified")
    phone: str = Field(description="Phone number where the code was sent")
    role: Literal["user", "employee"] = Field(
        description="Role that must present the code"
    )
    code: str = Field(description="Six digit verification code")
    verified: bool = Field(
        default=False, description="Whether the code has been verified"
    )


class HotelInfo(BaseModel):
    hotel_name: str = Field(description="Official hotel name")
    city: str = Field(description="City where the hotel is located")
    check_in_time: str = Field(description="Standard check-in time")
    check_out_time: str = Field(description="Standard check-out time")
    contact_phone: str = Field(description="Primary hotel phone number")
    breakfast_included: bool = Field(description="Whether breakfast is included")


class HotelCalleDB(DB):
    hotel_info: HotelInfo = Field(description="General information about the hotel")
    users: Dict[str, User] = Field(description="Guest profiles indexed by user id")
    room_types: Dict[str, RoomType] = Field(
        description="Room catalog indexed by room type id"
    )
    rooms: Dict[str, HotelRoom] = Field(
        description="Physical hotel rooms indexed by room id"
    )
    reservations: Dict[str, Reservation] = Field(
        description="Reservations indexed by reservation id"
    )
    verification_codes: Dict[str, VerificationCode] = Field(
        default_factory=dict,
        description="Active SMS verification codes indexed by reservation id",
    )

    def get_statistics(self) -> dict[str, Any]:
        return {
            "hotel_name": self.hotel_info.hotel_name,
            "num_room_types": len(self.room_types),
            "num_rooms": len(self.rooms),
            "num_reservations": len(self.reservations),
        }


def get_db() -> HotelCalleDB:
    return HotelCalleDB.load(HOTEL_CALLE_DB_PATH)
