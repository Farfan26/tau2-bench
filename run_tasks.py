import json
from tau2.domains.retail_farfan.environment import get_environment

# cargar entorno
env = get_environment()

# cargar tasks
with open("data/tau2/domains/retail_farfan/tasks.json") as f:
    tasks = json.load(f)

# simulación simple sin LLM
for task in tasks:
    print("\n============================")
    print("TASK:", task["id"])

    user_input = task["user_scenario"]["instructions"]["reason_for_call"]

    print("USER:", user_input)
    print("AGENT: (simulado) Procesando solicitud...")
