import os
import sys
import json
import time  # Para las pausas
import google.generativeai as genai

sys.path.append(os.getcwd())
genai.configure(api_key="AIzaSyBkyTdLtCfSd7R4p3ckLaissYgjCWwTUdE")
model = genai.GenerativeModel("models/gemini-3.1-flash-lite-preview")

from src.tau2.domains.retail_farfan.environment import get_environment, get_tasks


def evaluar_proyecto():
    env = get_environment()
    tasks = get_tasks("base")

    with open("data/tau2/domains/retail_farfan/db.json", "r", encoding="utf-8") as f:
        db_content = json.dumps(json.load(f), indent=2)

    total_tasks = len(tasks)
    exitos = 0

    print(f"🚀 Iniciando evaluación controlada de {total_tasks} tareas...")

    for t in tasks:
        instruccion = t["user_scenario"]["instructions"]["task_instructions"]
        contexto = t["user_scenario"]["instructions"].get("known_info", "N/A")
        esperado = t["evaluation_criteria"]["nl_assertions"][0]

        # PROMPT MEJORADO: Le decimos que sea directo
        prompt = f"""
        POLÍTICA: {env.policy}
        DB: {db_content}
        CONTEXTO: {contexto}
        TAREA: {instruccion}
        
        REGLA CRÍTICA: Responde de forma directa. Si la acción está permitida, 
        confirma la acción (ej: 'Pedido creado' o 'Pago procesado'). 
        Si no, explica por qué según la política.
        """

        try:
            # Ejecución de la tarea
            response = model.generate_content(prompt)
            respuesta_ia = response.text

            # Verificación: ¿Se cumple el objetivo?
            check_prompt = f"¿La respuesta del agente: '{respuesta_ia}' cumple con el objetivo esperado: '{esperado}'? Responde únicamente SI o NO."
            check = model.generate_content(check_prompt).text.strip().upper()

            status = "✅ PASÓ" if "SI" in check else "❌ FALLÓ"
            if "SI" in check:
                exitos += 1

            print(f"Tarea {t['id']} [{t['description']['purpose']}]: {status}")

            # PAUSA DE SEGURIDAD PARA LA CUOTA (4 segundos)
            time.sleep(4)

        except Exception as e:
            print(f"⚠️ Error en tarea {t['id']}: {e}")
            print("Esperando 10 segundos para reintentar por cuota...")
            time.sleep(10)

    eficacia = (exitos / total_tasks) * 100
    print("\n" + "=" * 40)
    print(f"📊 REPORTE FINAL DE RETAIL FARFÁN")
    print(f"Total: {total_tasks} | Exitos: {exitos}")
    print(f"EFICACIA FINAL: {eficacia}%")
    print("=" * 40)


evaluar_proyecto()
