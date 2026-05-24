import json
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from .data_model import DivemotorDB
from .tools import DivemotorTools


def get_environment(solo_mode=False):
    with open("data/tau2/domains/divemotor_santiago/db.json") as f:
        data = json.load(f)

    db = DivemotorDB(**data)
    tools = DivemotorTools(db=db)

    with open("data/tau2/domains/divemotor_santiago/policy.md") as f:
        policy = f.read()

    return Environment(
        domain_name="divemotor_santiago",
        policy=policy,
        tools=tools,
        solo_mode=solo_mode,
    )


def get_tasks(task_split_name="base"):
    with open("data/tau2/domains/divemotor_santiago/tasks.json") as f:
        tasks_data = json.load(f)

    with open("data/tau2/domains/divemotor_santiago/split_tasks.json") as f:
        splits = json.load(f)

    if task_split_name not in splits:
        selected = tasks_data
    else:
        selected_ids = set(splits[task_split_name])
        selected = [t for t in tasks_data if t["id"] in selected_ids]

    return [Task(**t) for t in selected]


def get_tasks_split():
    with open("data/tau2/domains/divemotor_santiago/split_tasks.json") as f:
        return json.load(f)
