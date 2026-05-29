from typing import Optional

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.utils import load_file

from tau2.domains.retail_farfan.data_model import RetailDB
from tau2.domains.retail_farfan.tools import RetailTools  # type: ignore
from tau2.domains.retail_farfan.utils import (
    RETAIL_DB_PATH,
    RETAIL_POLICY_PATH,
    RETAIL_TASK_SET_PATH,
    RETAIL_SPLIT_TASK_PATH,  # type: ignore
)


def get_environment(db: Optional[RetailDB] = None) -> Environment:
    """
    Inicializa y retorna el entorno para el dominio retail_farfan.
    Si no se provee una DB, la carga desde RETAIL_DB_PATH.
    """
    if db is None:
        db = RetailDB.load(RETAIL_DB_PATH)  # type: ignore

    tools = RetailTools(db)

    with open(RETAIL_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()

    return Environment(
        domain_name="retail_farfan",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """
    Carga las tareas desde tasks.json y las filtra según el split indicado.
    Por defecto usa el split 'base' que contiene todas las tareas.
    """
    tasks_data = load_file(RETAIL_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks_data]

    if task_split_name is None:
        return tasks

    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Split '{task_split_name}' no existe. "
            f"Splits disponibles: {list(task_splits.keys())}"
        )

    split_ids = [str(tid) for tid in task_splits[task_split_name]]
    return [task for task in tasks if str(task.id) in split_ids]


def get_tasks_split() -> dict[str, list[str]]:
    """
    Retorna el diccionario de splits de tareas desde split_tasks.json.
    """
    return load_file(RETAIL_SPLIT_TASK_PATH)
