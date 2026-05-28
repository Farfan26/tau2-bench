import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.hotel_calle.data_model import (
    HotelCalleDB,
    HotelInfo,
    HotelRoom,
    Reservation,
    RoomType,
    User,
)
from tau2.domains.hotel_calle.environment import get_environment
from tau2.environment.environment import Environment


@pytest.fixture
def hotel_db() -> HotelCalleDB:
    return HotelCalleDB(
        hotel_info=HotelInfo(
            hotel_name="Hotel Calle",
            city="Lima",
            check_in_time="15:00",
            check_out_time="12:00",
            contact_phone="+51 999 555 111",
            breakfast_included=True,
        ),
        users={},
        room_types={
            "standard_queen": RoomType(
                room_type_id="standard_queen",
                name="Standard Queen",
                description="Habitacion base",
                bed_configuration="1 cama queen",
                price_per_night=180.0,
                available_rooms=2,
                max_guests=2,
                amenities=["wifi", "desayuno"],
            )
        },
        rooms={
            "ROOM-001": HotelRoom(
                room_id="ROOM-001",
                room_number="101",
                room_type_id="standard_queen",
                floor=1,
                status="available",
                notes="",
            ),
            "ROOM-002": HotelRoom(
                room_id="ROOM-002",
                room_number="102",
                room_type_id="standard_queen",
                floor=1,
                status="available",
                notes="",
            ),
            "ROOM-003": HotelRoom(
                room_id="ROOM-003",
                room_number="103",
                room_type_id="standard_queen",
                floor=1,
                status="maintenance",
                notes="out of service",
            ),
        },
        reservations={},
    )


@pytest.fixture
def environment(hotel_db: HotelCalleDB) -> Environment:
    return get_environment(hotel_db)


def test_create_reservation(environment: Environment):
    response = environment.get_response(
        ToolCall(
            id="1",
            name="create_reservation",
            arguments={
                "guest_name": "Andrea Flores",
                "room_type_id": "standard_queen",
                "check_in_date": "2026-04-10",
                "check_out_date": "2026-04-12",
                "guests": 2,
                "special_request": "late arrival",
                "guest_email": "andrea.flores@example.com",
                "guest_phone": "+51 900 111 222",
            },
        )
    )
    assert not response.error
    reservation = environment.tools.db.reservations["RES-001"]
    assert reservation.user_id == "USER-001"
    assert reservation.guest_name == "Andrea Flores"
    assert reservation.guest_email == "andrea.flores@example.com"
    assert reservation.guest_phone == "+51 900 111 222"
    assert reservation.room_type_name == "Standard Queen"
    assert reservation.room_id == "ROOM-001"
    assert reservation.room_number == "101"
    assert reservation.total_price == 360.0
    assert environment.tools.db.users["USER-001"].full_name == "Andrea Flores"
    assert environment.tools.db.room_types["standard_queen"].available_rooms == 1
    assert environment.tools.db.rooms["ROOM-001"].status == "reserved"


def test_create_reservation_rejects_excess_guests(environment: Environment):
    response = environment.get_response(
        ToolCall(
            id="2",
            name="create_reservation",
            arguments={
                "guest_name": "Andrea Flores",
                "room_type_id": "standard_queen",
                "check_in_date": "2026-04-10",
                "check_out_date": "2026-04-12",
                "guests": 3,
                "special_request": "",
            },
        )
    )
    assert response.error


def test_create_reservation_allows_missing_contact_info(environment: Environment):
    response = environment.get_response(
        ToolCall(
            id="3",
            name="create_reservation",
            arguments={
                "guest_name": "Andrea Flores",
                "room_type_id": "standard_queen",
                "check_in_date": "2026-04-10",
                "check_out_date": "2026-04-12",
                "guests": 2,
                "special_request": "",
            },
        )
    )
    assert not response.error
    reservation = environment.tools.db.reservations["RES-001"]
    assert reservation.guest_email == ""
    assert reservation.guest_phone == ""


def test_check_room_availability_uses_dates_and_ignores_maintenance(
    environment: Environment,
):
    environment.tools.db.users["USER-900"] = User(
        user_id="USER-900",
        full_name="Existing Guest",
        email="existing@example.com",
        phone="+51 900 000 000",
    )
    environment.tools.db.reservations["RES-900"] = Reservation(
        reservation_id="RES-900",
        user_id="USER-900",
        guest_name="Existing Guest",
        guest_email="existing@example.com",
        guest_phone="+51 900 000 000",
        room_type_id="standard_queen",
        room_type_name="Standard Queen",
        room_id="ROOM-001",
        room_number="101",
        check_in_date="2026-04-10",
        check_out_date="2026-04-12",
        guests=2,
        total_price=360.0,
        special_request="",
        status="confirmed",
    )
    environment.tools.db.rooms["ROOM-001"].status = "reserved"

    overlapping = environment.use_tool(
        "check_room_availability",
        room_type_id="standard_queen",
        check_in_date="2026-04-10",
        check_out_date="2026-04-12",
        guests=2,
    )
    assert overlapping["available_rooms"] == 1

    non_overlapping = environment.use_tool(
        "check_room_availability",
        room_type_id="standard_queen",
        check_in_date="2026-04-12",
        check_out_date="2026-04-14",
        guests=2,
    )
    assert non_overlapping["available_rooms"] == 2


def test_sms_verification_flow(environment: Environment):
    environment.get_response(
        ToolCall(
            id="sms-setup",
            name="create_reservation",
            arguments={
                "guest_name": "Andrea Flores",
                "room_type_id": "standard_queen",
                "check_in_date": "2026-04-10",
                "check_out_date": "2026-04-12",
                "guests": 2,
                "special_request": "",
                "guest_phone": "+51 900 111 222",
            },
        )
    )

    send_response = environment.get_response(
        ToolCall(
            id="sms-send",
            name="send_sms_verification_code",
            arguments={"reservation_id": "RES-001", "role": "user"},
        )
    )
    assert not send_response.error

    code = environment.use_user_tool(
        "read_latest_sms_code", reservation_id="RES-001"
    )
    assert code.isdigit()
    assert len(code) == 6

    verify_response = environment.get_response(
        ToolCall(
            id="sms-verify",
            name="verify_sms_code",
            arguments={
                "reservation_id": "RES-001",
                "role": "user",
                "code": code,
            },
        )
    )
    assert not verify_response.error
    assert environment.tools.assert_sms_verification_status("RES-001", True)


def test_sms_verification_rejects_wrong_code(environment: Environment):
    environment.get_response(
        ToolCall(
            id="sms-setup",
            name="create_reservation",
            arguments={
                "guest_name": "Andrea Flores",
                "room_type_id": "standard_queen",
                "check_in_date": "2026-04-10",
                "check_out_date": "2026-04-12",
                "guests": 2,
                "special_request": "",
                "guest_phone": "+51 900 111 222",
            },
        )
    )
    environment.get_response(
        ToolCall(
            id="sms-send",
            name="send_sms_verification_code",
            arguments={"reservation_id": "RES-001", "role": "user"},
        )
    )

    result = environment.use_tool(
        "verify_sms_code",
        reservation_id="RES-001",
        role="user",
        code="000000",
    )
    assert result["verified"] is False
    assert result["reason"] == "incorrect_code"
    assert environment.tools.assert_sms_verification_status("RES-001", False)
