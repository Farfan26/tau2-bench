import json
from tau2.environment import Environment
from .data_model import DivemotorDB
from .tools import DivemotorTools


def get_environment():
    with open("data/tau2/domains/divemotor_santiago/db.json") as f:
        data = json.load(f)

    db = DivemotorDB(**data)
    tools = DivemotorTools(db=db)

    return Environment(db=db, toolkit=tools)


def get_tasks():
    with open("data/tau2/domains/divemotor_santiago/tasks.json") as f:
        return json.load(f)


def get_tasks_split():
    with open("data/tau2/domains/divemotor_santiago/split_tasks.json") as f:
        return json.load(f)