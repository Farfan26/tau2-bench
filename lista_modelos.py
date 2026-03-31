import google.generativeai as genai

# Tu clave
genai.configure(api_key="AIzaSyBkyTdLtCfSd7R4p3ckLaissYgjCWwTUdE")

print("--- Lista de modelos disponibles ---")
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error al conectar: {e}")
