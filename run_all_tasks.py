import json
from tau2.domains.retail_farfan.environment import get_environment

# cargar entorno
env = get_environment()

# cargar tasks
with open("data/tau2/domains/retail_farfan/tasks.json", "r", encoding="utf-8") as f:
    tasks = json.load(f)

print("\n🚀 EJECUTANDO TASKS DEL DOMINIO retail_farfan\n")

for task in tasks:
    print("\n" + "=" * 50)
    print("TASK:", task["id"])

    scenario = task["user_scenario"]["instructions"]

    user_input = scenario["reason_for_call"]
    known_info = scenario["known_info"]

    print("USER:", user_input)
    print("INFO:", known_info)

    try:
        # simular interacción con el entorno
        result = env.step(user_input)

        print("AGENT:", result)

    except Exception as e:
        print("ERROR:", str(e))
