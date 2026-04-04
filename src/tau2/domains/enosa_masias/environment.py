from typing import Optional
from tau2.environment.environment import Environment
from tau2.data_model.tasks import Task
from tau2.domains.enosa_masias.data_model import EnosaDB
from tau2.domains.enosa_masias.tools import EnosaToolKit
from tau2.domains.enosa_masias.utils import (
    ENOSA_DB_PATH, ENOSA_TASKS_PATH, ENOSA_SPLIT_TASKS_PATH, ENOSA_POLICY_PATH
)
import json

def get_environment(db: Optional[EnosaDB] = None, solo_mode: bool = False) -> Environment:
    if solo_mode:
        raise ValueError("ENOSA domain does not support solo mode")
    if db is None:
        db = EnosaDB.load()
    policy = ENOSA_POLICY_PATH.read_text(encoding="utf-8")
    toolkit = EnosaToolKit(db=db)
    return Environment(domain_name="enosa_masias", policy=policy, tools=toolkit)

def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks_data = json.loads(ENOSA_TASKS_PATH.read_text(encoding="utf-8"))
    splits = get_tasks_split()
    task_ids = splits.get(task_split_name, [])
    return [Task.model_validate(t) for t in tasks_data if t["id"] in task_ids]

def get_tasks_split() -> dict:
    return json.loads(ENOSA_SPLIT_TASKS_PATH.read_text(encoding="utf-8"))