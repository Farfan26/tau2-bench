import json
import os
from tau2.environment.environment import Environment


def get_environment():
    """
    Inicializa el entorno de Tau2 para el dominio retail_farfan.
    """
    domain_name = "retail_farfan"

    # Definimos la ruta de la política manualmente para evitar el ImportError
    # Suponiendo que estás ejecutando desde la raíz de tau2-bench
    policy_path = os.path.join("data", "tau2", "domains", domain_name, "policy.md")

    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            policy_content = f.read()
    except FileNotFoundError:
        policy_content = "Responde a las solicitudes del usuario de manera profesional."

    # Pasamos los argumentos requeridos
    env = Environment(domain_name=domain_name, policy=policy_content)
    return env


def get_tasks(task_split_name="base"):
    """
    Carga las tasks y filtra según el split definido.
    """
    base_path = os.path.join("data", "tau2", "domains", "retail_farfan")

    with open(os.path.join(base_path, "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)

    with open(os.path.join(base_path, "split_tasks.json"), encoding="utf-8") as f:
        splits = json.load(f)

    ids = splits.get(task_split_name, [])
    return [t for t in tasks if t["id"] in ids]


def get_tasks_split():
    """
    Retorna todos los splits disponibles.
    """
    base_path = os.path.join("data", "tau2", "domains", "retail_farfan")
    with open(os.path.join(base_path, "split_tasks.json"), encoding="utf-8") as f:
        return json.load(f)
