from pathlib import Path
import os

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]

if "TAU2_DATA_DIR" in os.environ:
    ENOSA_DATA_DIR = Path(os.environ["TAU2_DATA_DIR"]) / "domains" / "enosa_masias"
else:
    ENOSA_DATA_DIR = _REPO_ROOT / "data" / "tau2" / "domains" / "enosa_masias"

ENOSA_DB_PATH = ENOSA_DATA_DIR / "db.json"
ENOSA_TASKS_PATH = ENOSA_DATA_DIR / "tasks.json"
ENOSA_SPLIT_TASKS_PATH = ENOSA_DATA_DIR / "split_tasks.json"
ENOSA_POLICY_PATH = _THIS_FILE.parent / "policy.md"

if __name__ == "__main__":
    from tau2.domains.enosa_masias.data_model import EnosaDB, User, Supply, Ticket

    db = EnosaDB(
        users={
            "48912304": User(user_id="48912304", name="Karla Chero", phone="987", email="karla@gmail.com"),
            "76543210": User(user_id="76543210", name="Martin Masias", phone="912", email="martin@outlook.com"),
            "02847193": User(user_id="02847193", name="Juan Perez", phone="966", email="jperez@yahoo.com"),
            "45123987": User(user_id="45123987", name="Maria Gonzales", phone="944", email="mg@gmail.com"),
            "70112233": User(user_id="70112233", name="Luis Zapata", phone="998", email="luis@hotmail.com")
        },
        supplies={
            "S-1001": Supply(supply_number="S-1001", owner_id="48912304", address="Av Grau", status="disconnected_due_to_debt", debt_amount=150.0),
            "S-1002": Supply(supply_number="S-1002", owner_id="76543210", address="Urb Miraflores", status="active", debt_amount=0.0),
            "S-1003": Supply(supply_number="S-1003", owner_id="02847193", address="Jr Comercio", status="active", debt_amount=0.0),
            "S-1004": Supply(supply_number="S-1004", owner_id="45123987", address="Av Loreto", status="active", debt_amount=0.0),
            "S-1005": Supply(supply_number="S-1005", owner_id="70112233", address="Zona Industrial", status="active", debt_amount=0.0)
        },
        tickets={
            "T001": Ticket(ticket_id="T001", reporter_id="48912304", supply_number="S-1001", issue_type="billing", description="High bill", status="in_progress", creation_date="2026-04-01")
        }
    )

    ENOSA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    db.dump(ENOSA_DB_PATH)
    print(f"db.json generated at: {ENOSA_DB_PATH}")