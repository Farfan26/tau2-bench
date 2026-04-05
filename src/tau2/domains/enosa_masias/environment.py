from typing import Optional
from tau2.data_model.tasks import Task
from tau2.domains.enosa_masias.data_model import EnosaDB
from tau2.domains.enosa_masias.tools import EnosaToolKit
from tau2.domains.enosa_masias.utils import (
    ENOSA_DB_PATH, ENOSA_POLICY_PATH, ENOSA_TASK_SPLIT_PATH, ENOSA_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file

def get_environment(db: Optional[EnosaDB] = None, solo_mode: bool = False) -> Environment:
    if solo_mode: raise ValueError("ENOSA no soporta solo_mode")
    if db is None: db = EnosaDB.load(ENOSA_DB_PATH)
    
    tools = EnosaToolKit(db)
    with open(ENOSA_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()
        
    return Environment(domain_name="enosa_masias", policy=policy, tools=tools)

def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(ENOSA_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    
    if task_split_name is None: return tasks
    
    task_splits = get_tasks_split()
    # Aquí validamos si el split (train, test, base) existe
    if task_split_name not in task_splits:
        raise ValueError(f"Split '{task_split_name}' no encontrado. Usa: {list(task_splits.keys())}")
        
    return [t for t in tasks if t.id in task_splits[task_split_name]]

def get_tasks_split() -> dict[str, list[str]]:
    return load_file(ENOSA_TASK_SPLIT_PATH)