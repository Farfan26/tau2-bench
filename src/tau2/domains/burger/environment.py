from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.burger.data_model import BurgerDB
from tau2.domains.burger.tools import BurgerTools
from tau2.domains.burger.utils import (
    BURGER_DB_PATH,
    BURGER_POLICY_PATH,
    BURGER_POLICY_RAG_PATH,
    BURGER_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.environment.rag import THINK_INSTRUCTION, ChromaPolicyIndex
from tau2.utils import load_file


def get_environment(
    db: Optional[BurgerDB] = None,
    solo_mode: bool = False,
    chunking_strategy: str = "headers",
    retrieval_k: int = 3,
    use_think: bool = False,
    use_rag: bool = True,
) -> Environment:
    if solo_mode:
        raise ValueError("Burger domain does not support solo mode")
    if db is None:
        db = BurgerDB.load(BURGER_DB_PATH)

    if use_rag:
        with open(BURGER_POLICY_PATH, "r", encoding="utf-8") as fp:
            policy_text = fp.read()
        policy_index = ChromaPolicyIndex(policy_text, strategy=chunking_strategy)
        tools = BurgerTools(db, policy_index=policy_index, retrieval_k=retrieval_k)
        with open(BURGER_POLICY_RAG_PATH, "r", encoding="utf-8") as fp:
            policy = fp.read()
        if use_think:
            policy = policy + THINK_INSTRUCTION
    else:
        tools = BurgerTools(db)
        with open(BURGER_POLICY_PATH, "r", encoding="utf-8") as fp:
            policy = fp.read()

    return Environment(
        domain_name="burger",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(BURGER_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(BURGER_TASK_SET_PATH).parent
        / f"split_{Path(BURGER_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
