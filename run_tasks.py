import json
from tau2.domains.retail_farfan.environment import get_environment
from tau2.agent.llm_agent import LLMAgent

# cargar entorno
env = get_environment()

# inicializar agente
agent = LLMAgent()

# cargar tasks
with open("data/tau2/domains/retail_farfan/tasks.json") as f:
    tasks = json.load(f)

# ejecutar tasks
for task in tasks:
    print("\n============================")
    print("TASK:", task["id"])

    user_input = task["user_scenario"]["instructions"]["reason_for_call"]

    response = agent.chat(user_input, env=env)

    print("USER:", user_input)
    print("AGENT:", response)
