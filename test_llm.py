import os
import sys
import json
import google.generativeai as genai

# 1. Configuración de rutas y API Key
sys.path.append(os.getcwd())

# Configuración de la API con tu clave directa
API_KEY = "AIzaSyBkyTdLtCfSd7R4p3ckLaissYgjCWwTUdE"
genai.configure(api_key=API_KEY)

# USAMOS EL NOMBRE EXACTO QUE SALIÓ EN TU LISTA
MODEL_NAME = "models/gemini-3.1-flash-lite-preview"
model = genai.GenerativeModel(MODEL_NAME)

# 2. Cargar tus archivos de Retail Farfán
try:
    from src.tau2.domains.retail_farfan.environment import get_environment, get_tasks

    env = get_environment()
    tasks = get_tasks("base")

    # Cargamos la base de datos para darle contexto real a la IA
    db_path = "data/tau2/domains/retail_farfan/db.json"
    with open(db_path, "r", encoding="utf-8") as f:
        db_data = json.load(f)
        db_content = json.dumps(db_data, indent=2, ensure_ascii=False)

except Exception as e:
    print(f"❌ Error al cargar archivos de dominio o DB: {e}")
    sys.exit(1)

# 3. Ejecución de la Tarea 1
if not tasks:
    print("❌ No se encontraron tareas en el archivo JSON.")
else:
    tarea = tasks[0]

    # Extraemos la información según la estructura de tu JSON
    try:
        instrucciones_bloque = tarea["user_scenario"]["instructions"]
        instruccion_tarea = instrucciones_bloque["task_instructions"]
        contexto_usuario = instrucciones_bloque.get("known_info", "N/A")
        objetivo_tarea = tarea["description"]["purpose"]
    except KeyError as e:
        print(f"❌ Error en la estructura del JSON: falta la llave {e}")
        sys.exit(1)

    print(f"--- Probando Tarea ID: {tarea['id']} ---")
    print(f"Objetivo: {objetivo_tarea}")
    print(f"Instrucción: {instruccion_tarea}")
    print(f"Contexto: {contexto_usuario}")

    # 4. Prompt Maestro optimizado para Retail Farfán
    prompt = f"""
    Eres el asistente virtual oficial de RETAIL_FARFAN.
    
    POLÍTICA DE NEGOCIO:
    {env.policy}
    
    BASE DE DATOS (Estado actual):
    {db_content}
    
    CASO A RESOLVER:
    - Contexto: {contexto_usuario}
    - Tarea: {instruccion_tarea}
    
    INSTRUCCIONES:
    Responde al usuario de forma profesional y en español. 
    Verifica siempre en la DB si el usuario está activo y si hay stock del producto solicitado.
    """

    print(f"\n⏳ Consultando a {MODEL_NAME}...")
    try:
        response = model.generate_content(prompt)
        print("\n--- Respuesta del LLM ---")
        print(response.text)
    except Exception as e:
        print(f"❌ Error en la generación: {e}")
