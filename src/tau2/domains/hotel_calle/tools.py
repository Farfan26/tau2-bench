from datetime import date

from tau2.domains.hotel_calle.data_model import (
    HotelCalleDB,
    HotelRoom,
    Reservation,
    RoomType,
    User,
    VerificationCode,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HotelCalleTools(ToolKitBase):
    """Tools for a simple hotel reception chatbox."""

    db: HotelCalleDB

    def __init__(self, db: HotelCalleDB) -> None:
        super().__init__(db)

    def _get_room_type(self, room_type_id: str) -> RoomType:
        if room_type_id not in self.db.room_types:
            raise ValueError(f"Room type '{room_type_id}' not found")
        return self.db.room_types[room_type_id]

    def _parse_dates(self, check_in_date: str, check_out_date: str) -> tuple[date, date]:
        try:
            check_in = date.fromisoformat(check_in_date)
            check_out = date.fromisoformat(check_out_date)
        except ValueError as exc:
            raise ValueError("Dates must use YYYY-MM-DD format") from exc
        if check_out <= check_in:
            raise ValueError("Check-out date must be after check-in date")
        return check_in, check_out

    def _dates_overlap(
        self,
        start_a: date,
        end_a: date,
        start_b: date,
        end_b: date,
    ) -> bool:
        return start_a < end_b and start_b < end_a

    def _reservation_blocks_inventory(self, reservation: Reservation) -> bool:
        return reservation.status in {"confirmed", "pending"}

    def _generate_sms_code(self, reservation_id: str, role: str) -> str:
        raw = f"{reservation_id}:{role}:{self.db.reservations[reservation_id].guest_phone}"
        checksum = sum((idx + 1) * ord(char) for idx, char in enumerate(raw))
        return f"{checksum % 1000000:06d}"

    def _get_available_rooms_by_type(
        self,
        room_type_id: str,
        check_in_date: str,
        check_out_date: str,
    ) -> list[HotelRoom]:
        check_in, check_out = self._parse_dates(check_in_date, check_out_date)
        blocked_room_ids: set[str] = set()

        for reservation in self.db.reservations.values():
            if not self._reservation_blocks_inventory(reservation):
                continue
            if reservation.room_id is None:
                continue
            reservation_check_in, reservation_check_out = self._parse_dates(
                reservation.check_in_date,
                reservation.check_out_date,
            )
            if self._dates_overlap(
                check_in,
                check_out,
                reservation_check_in,
                reservation_check_out,
            ):
                blocked_room_ids.add(reservation.room_id)

        return [
            room
            for room in self.db.rooms.values()
            if room.room_type_id == room_type_id
            and room.status != "maintenance"
            and room.room_id not in blocked_room_ids
        ]

    def _find_or_create_user(
        self,
        guest_name: str,
        guest_email: str = "",
        guest_phone: str = "",
    ) -> User:
        normalized_name = guest_name.strip()
        normalized_email = guest_email.strip()
        normalized_phone = guest_phone.strip()

        for user in self.db.users.values():
            same_name = user.full_name == normalized_name
            same_email = not normalized_email or user.email == normalized_email
            same_phone = not normalized_phone or user.phone == normalized_phone
            if same_name and same_email and same_phone:
                if normalized_email and not user.email:
                    user.email = normalized_email
                if normalized_phone and not user.phone:
                    user.phone = normalized_phone
                return user

        user_id = f"USER-{len(self.db.users) + 1:03d}"
        user = User(
            user_id=user_id,
            full_name=normalized_name,
            email=normalized_email,
            phone=normalized_phone,
        )
        self.db.users[user_id] = user
        return user

    @is_tool(ToolType.READ)
    def get_hotel_info(self) -> dict:
        """
        Get the core hotel information that an assistant can share with guests.
        """
        info = self.db.hotel_info
        return {
            "hotel_name": info.hotel_name,
            "city": info.city,
            "check_in_time": info.check_in_time,
            "check_out_time": info.check_out_time,
            "contact_phone": info.contact_phone,
            "breakfast_included": info.breakfast_included,
        }

    @is_tool(ToolType.READ)
    def get_room_catalog(self) -> list[RoomType]:
        """
        List every room type the hotel offers, including amenities and prices.
        """
        return list(self.db.room_types.values())

    @is_tool(ToolType.READ)
    def check_room_availability(
        self, room_type_id: str, check_in_date: str, check_out_date: str, guests: int
    ) -> dict:
        """
        Check whether a room type can host the requested stay.

        Args:
            room_type_id: Room type identifier such as 'doble'.
            check_in_date: Check-in date in YYYY-MM-DD format.
            check_out_date: Check-out date in YYYY-MM-DD format.
            guests: Number of guests who will stay in the room.

        Returns:
            Availability details including nights and estimated total.
        """
        if guests < 1:
            raise ValueError("Guests must be at least 1")
        check_in, check_out = self._parse_dates(check_in_date, check_out_date)
        room_type = self._get_room_type(room_type_id)
        available_rooms = self._get_available_rooms_by_type(
            room_type_id=room_type_id,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
        )
        nights = (check_out - check_in).days
        return {
            "room_type_id": room_type.room_type_id,
            "room_type_name": room_type.name,
            "available": len(available_rooms) > 0 and guests <= room_type.max_guests,
            "available_rooms": len(available_rooms),
            "nights": nights,
            "max_guests": room_type.max_guests,
            "estimated_total": round(room_type.price_per_night * nights, 2),
        }

    @is_tool(ToolType.WRITE)
    def create_reservation(
        self,
        guest_name: str,
        room_type_id: str,
        check_in_date: str,
        check_out_date: str,
        guests: int,
        special_request: str = "",
        guest_email: str = "",
        guest_phone: str = "",
    ) -> Reservation:
        """
        Create a hotel reservation if the room type is available.

        Args:
            guest_name: Full name of the guest.
            room_type_id: Room type identifier.
            check_in_date: Check-in date in YYYY-MM-DD format.
            check_out_date: Check-out date in YYYY-MM-DD format.
            guests: Number of guests included in the reservation.
            special_request: Optional note such as 'late arrival'.
            guest_email: Optional guest email address collected during the booking flow.
            guest_phone: Optional guest phone number collected during the booking flow.
        """
        if not guest_name.strip():
            raise ValueError("Guest name must not be empty")
        if guests < 1:
            raise ValueError("Guests must be at least 1")

        availability = self.check_room_availability(
            room_type_id=room_type_id,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            guests=guests,
        )
        if not availability["available"]:
            raise ValueError("Requested room is not available for this reservation")

        room_type = self._get_room_type(room_type_id)
        assigned_room = self._get_available_rooms_by_type(
            room_type_id=room_type_id,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
        )[0]
        user = self._find_or_create_user(
            guest_name=guest_name,
            guest_email=guest_email,
            guest_phone=guest_phone,
        )
        reservation_id = f"RES-{len(self.db.reservations) + 1:03d}"
        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user.user_id,
            guest_name=guest_name.strip(),
            guest_email=guest_email.strip(),
            guest_phone=guest_phone.strip(),
            room_type_id=room_type.room_type_id,
            room_type_name=room_type.name,
            room_id=assigned_room.room_id,
            room_number=assigned_room.room_number,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            guests=guests,
            total_price=availability["estimated_total"],
            special_request=special_request.strip(),
            status="confirmed",
        )
        self.db.reservations[reservation_id] = reservation
        assigned_room.status = "reserved"
        room_type.available_rooms = len(
            [
                room
                for room in self.db.rooms.values()
                if room.room_type_id == room_type_id and room.status == "available"
            ]
        )
        return reservation

    @is_tool(ToolType.READ)
    def get_reservation(self, reservation_id: str) -> Reservation:
        """
        Retrieve a reservation using its identifier.
        """
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation '{reservation_id}' not found")
        return self.db.reservations[reservation_id]

    @is_tool(ToolType.WRITE)
    def send_sms_verification_code(
        self, reservation_id: str, role: str = "user"
    ) -> dict:
        """
        Send a verification code by SMS for a sensitive reservation operation.

        Args:
            reservation_id: Reservation identifier to verify.
            role: Role being verified. Must be 'user' or 'employee'.
        """
        if role not in {"user", "employee"}:
            raise ValueError("Role must be 'user' or 'employee'")
        reservation = self.get_reservation(reservation_id)
        if not reservation.guest_phone:
            raise ValueError("Reservation does not have a phone number for SMS")

        code = self._generate_sms_code(reservation_id, role)
        self.db.verification_codes[reservation_id] = VerificationCode(
            reservation_id=reservation_id,
            phone=reservation.guest_phone,
            role=role,
            code=code,
            verified=False,
        )
        return {
            "reservation_id": reservation_id,
            "role": role,
            "sent": True,
            "phone_last_digits": reservation.guest_phone[-3:],
        }

    @is_tool(ToolType.WRITE)
    def verify_sms_code(
        self, reservation_id: str, code: str, role: str = "user"
    ) -> dict:
        """
        Verify an SMS code before a sensitive reservation operation.

        Args:
            reservation_id: Reservation identifier being verified.
            code: Code provided by the user.
            role: Role being verified. Must match the role used when sending the code.
        """
        if reservation_id not in self.db.verification_codes:
            raise ValueError("No SMS verification code has been sent for this reservation")
        verification = self.db.verification_codes[reservation_id]
        if verification.role != role:
            raise ValueError("Verification role does not match")
        if verification.code != code.strip():
            return {
                "reservation_id": reservation_id,
                "role": role,
                "verified": False,
                "reason": "incorrect_code",
            }
        verification.verified = True
        return {
            "reservation_id": reservation_id,
            "role": role,
            "verified": True,
        }

    def assert_reservation_status(self, reservation_id: str, expected_status: str) -> bool:
        """
        Check whether a reservation exists with the expected status.
        """
        reservation = self.get_reservation(reservation_id)
        return reservation.status == expected_status

    def assert_sms_verification_status(
        self, reservation_id: str, expected_verified: bool
    ) -> bool:
        """
        Check whether a reservation has the expected SMS verification status.
        """
        verification = self.db.verification_codes.get(reservation_id)
        if verification is None:
            return expected_verified is False
        return verification.verified == expected_verified
