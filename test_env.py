from tau2.domains.retail_farfan.environment import get_environment

# Dentro de test_env.py (o el script de prueba que estés usando)
domain_name = "retail_farfan"
env = get_environment()
print(env)
from tau2.domains.retail_farfan.environment import get_environment, get_tasks

# 1. Inicializamos el entorno
env = get_environment()
print("✅ Entorno cargado con éxito")

# 2. Cargamos las tareas
tasks = get_tasks("base")
print(f"✅ Se han cargado {len(tasks)} tareas para el dominio Retail Farfán")

# ... (código anterior)

# 3. Ver la primera tarea de forma segura
if tasks:
    primera_tarea = tasks[0]
    # Imprimimos las llaves para saber cómo se llaman (ej: id, task, etc.)
    print(f"Campos disponibles en la tarea: {list(primera_tarea.keys())}")

    # Intentamos buscar el contenido de la tarea
    # Cambia 'instruction' por 'task' si es necesario
    contenido = (
        primera_tarea.get("instruction")
        or primera_tarea.get("task")
        or "No se encontró el texto de la tarea"
    )

    print(f"✅ Contenido de la tarea: {contenido}")
