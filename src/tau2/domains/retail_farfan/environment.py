from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.utils import load_file

from tau2.domains.retail_farfan.data_model import RetailDB
from tau2.domains.retail_farfan.tools import RetailTools
from tau2.domains.retail_farfan.utils import (
    RETAIL_DB_PATH,
    RETAIL_POLICY_PATH,
    RETAIL_TASK_SET_PATH,
)


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
) -> Environment:
    """
    Inicializa el entorno para el dominio retail_farfan.
    """
    if db is None:
        db = RetailDB.load(RETAIL_DB_PATH)

    tools = RetailTools(db)

    with open(RETAIL_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()

    env = Environment(
        domain_name="retail_farfan",
        policy=policy,
        tools=tools,
    )

    if solo_mode:
        env.solo_mode = True

    return env


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """
    Carga las tareas usando load_file y filtra según el split.
    """
    tasks_data = load_file(RETAIL_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks_data]

    if task_split_name is None:
        return tasks

    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )

    # Nos aseguramos de comparar strings por si los IDs vienen como enteros en el JSON
    split_ids = [str(tid) for tid in task_splits[task_split_name]]
    return [task for task in tasks if str(task.id) in split_ids]


def get_tasks_split() -> dict[str, list[str]]:
    """
    Retorna el diccionario de splits disponible armando la ruta dinámicamente.
    """
    split_file = (
        Path(RETAIL_TASK_SET_PATH).parent
        / f"split_{Path(RETAIL_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
