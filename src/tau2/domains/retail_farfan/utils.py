from pathlib import Path

# ------------------------------------------------------------------
# Raíz del dominio en datos
# ------------------------------------------------------------------
DOMAIN_NAME = "retail_farfan"

# Directorio base de datos del dominio
# Asume la estructura estándar: data/tau2/domains/<nombre_dominio>/
DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "data"
    / "tau2"
    / "domains"
    / DOMAIN_NAME
)

# ------------------------------------------------------------------
# Rutas de archivos de datos
# ------------------------------------------------------------------

# Base de datos simulada
RETAIL_DB_PATH = DATA_DIR / "db.json"

# Política del agente (system prompt)
RETAIL_POLICY_PATH = DATA_DIR / "policy.md"

# Tareas de evaluación
RETAIL_TASK_SET_PATH = DATA_DIR / "tasks.json"

# Splits de tareas (base / train / test)
RETAIL_SPLIT_TASK_PATH = DATA_DIR / "split_tasks.json"
