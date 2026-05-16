#!/usr/bin/env python3
"""
Validates a student PR submission for Tau2-Bench Entrega 1.

Usage:
    python validate_student_pr.py --changed-files file1 file2 ...

Exit codes:
    0  All checks passed (or only warnings)
    1  One or more required checks failed
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

# Files required by the spec (utils.py is required, user_data_model.py and user_tools.py are optional)
REQUIRED_PYTHON_FILES = ["__init__.py", "data_model.py", "tools.py", "environment.py", "utils.py"]
OPTIONAL_PYTHON_FILES = ["user_data_model.py", "user_tools.py"]
REQUIRED_DATA_FILES = ["db.json", "tasks.json", "split_tasks.json", "policy.md"]

MIN_TASKS = 10
MAX_TASKS = 20
MIN_TOOLS = 5

# Known built-in domains — excluded from student domain detection
BUILTIN_DOMAINS = {"airline", "retail", "burger", "telecom", "mock"}


class CheckResult:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failed = []

    def ok(self, msg):
        self.passed.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def fail(self, msg):
        self.failed.append(msg)

    @property
    def success(self):
        return len(self.failed) == 0


def detect_domain(changed_files: list[str]) -> str | None:
    """Infer the domain name from the changed files list."""
    pattern = re.compile(r"src/tau2/domains/([^/]+)/")
    domains = set()
    for f in changed_files:
        m = pattern.match(f)
        if m:
            name = m.group(1)
            if name not in ("__pycache__",):
                domains.add(name)
    student_domains = domains - BUILTIN_DOMAINS
    if len(student_domains) == 1:
        return student_domains.pop()
    if len(student_domains) > 1:
        return None  # ambiguous — multiple student domains in one PR
    return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_module(domain: str, r: CheckResult):
    """Check all required Python files are present in the domain module."""
    base = REPO_ROOT / "src" / "tau2" / "domains" / domain
    for fname in REQUIRED_PYTHON_FILES:
        path = base / fname
        if path.exists():
            r.ok(f"`{fname}` presente en el módulo Python")
        else:
            r.fail(f"Falta `src/tau2/domains/{domain}/{fname}`")

    for fname in OPTIONAL_PYTHON_FILES:
        path = base / fname
        if path.exists():
            r.ok(f"`{fname}` presente (opcional, buen trabajo)")


def check_data_model_content(domain: str, r: CheckResult):
    """Check data_model.py defines a class that inherits from DB."""
    path = REPO_ROOT / "src" / "tau2" / "domains" / domain / "data_model.py"
    if not path.exists():
        return  # already flagged by check_python_module
    content = path.read_text()
    # Look for a class that inherits from DB (e.g. class HotelDB(DB):)
    if re.search(r"class\s+\w+\(DB\)", content):
        r.ok("`data_model.py` define una clase que hereda de `DB`")
    else:
        r.fail(
            "`data_model.py` no define ninguna clase que herede de `DB` — "
            "la clase principal de base de datos debe ser `class MiDominioDB(DB):`"
        )


def check_environment_content(domain: str, r: CheckResult):
    """Check environment.py defines the three required functions."""
    path = REPO_ROOT / "src" / "tau2" / "domains" / domain / "environment.py"
    if not path.exists():
        return
    content = path.read_text()
    for fn in ["get_environment", "get_tasks", "get_tasks_split"]:
        if re.search(rf"^def {fn}\b", content, re.MULTILINE):
            r.ok(f"`environment.py` define `{fn}()`")
        else:
            r.fail(
                f"`environment.py` no define `{fn}()` — "
                "esta función es requerida por el framework"
            )


def check_data_files(domain: str, r: CheckResult):
    """Check all required data files are present."""
    base = REPO_ROOT / "data" / "tau2" / "domains" / domain
    for fname in REQUIRED_DATA_FILES:
        path = base / fname
        if path.exists():
            r.ok(f"`{fname}` presente en archivos de datos")
        else:
            r.fail(f"Falta `data/tau2/domains/{domain}/{fname}`")


def check_db_json(domain: str, r: CheckResult):
    """Validate db.json is valid JSON and has at least 5 users."""
    db_path = REPO_ROOT / "data" / "tau2" / "domains" / domain / "db.json"
    if not db_path.exists():
        return
    try:
        db = json.loads(db_path.read_text())
    except json.JSONDecodeError as e:
        r.fail(f"`db.json` tiene JSON inválido: {e}")
        return
    r.ok("`db.json` es JSON válido")

    # Check for at least 5 users
    users = db.get("users", {})
    if isinstance(users, dict):
        n_users = len(users)
    elif isinstance(users, list):
        n_users = len(users)
    else:
        n_users = 0

    if n_users >= 5:
        r.ok(f"`db.json` contiene {n_users} usuarios (mínimo 5 requeridos)")
    elif n_users > 0:
        r.fail(
            f"`db.json` tiene solo {n_users} usuario(s) — "
            "se requieren al menos 5 usuarios con perfiles distintos"
        )
    else:
        r.warn(
            '`db.json` no tiene clave "users" verificable — '
            "asegúrate de incluir al menos 5 usuarios distintos"
        )


def check_tasks_json(domain: str, r: CheckResult):
    """Validate tasks.json: count, JSON validity, and required fields per task."""
    tasks_path = REPO_ROOT / "data" / "tau2" / "domains" / domain / "tasks.json"
    if not tasks_path.exists():
        return
    try:
        tasks = json.loads(tasks_path.read_text())
    except json.JSONDecodeError as e:
        r.fail(f"`tasks.json` tiene JSON inválido: {e}")
        return
    if not isinstance(tasks, list):
        r.fail("`tasks.json` debe ser una lista de tareas")
        return

    n = len(tasks)
    if MIN_TASKS <= n <= MAX_TASKS:
        r.ok(f"`tasks.json` contiene {n} tareas ({MIN_TASKS}–{MAX_TASKS} requeridas)")
    elif n < MIN_TASKS:
        r.fail(f"`tasks.json` tiene solo {n} tareas — se requieren al menos {MIN_TASKS}")
    else:
        r.warn(f"`tasks.json` tiene {n} tareas — el máximo recomendado es {MAX_TASKS}")

    # All four top-level fields are required per spec schema
    required_fields = {"id", "description", "user_scenario", "evaluation_criteria"}
    first_error = True
    for i, task in enumerate(tasks):
        missing = required_fields - set(task.keys())
        if missing and first_error:
            r.fail(
                f"Tarea #{i} (id={task.get('id', '?')}) en `tasks.json` "
                f"le faltan campos requeridos: {sorted(missing)}"
            )
            first_error = False  # report only first to avoid flooding

    if first_error:  # no errors found above
        r.ok("`tasks.json` — todas las tareas tienen los campos requeridos (`id`, `description`, `user_scenario`, `evaluation_criteria`)")


def check_split_tasks(domain: str, r: CheckResult):
    """Validate split_tasks.json: requires 'base', 'train', and 'test' splits."""
    split_path = REPO_ROOT / "data" / "tau2" / "domains" / domain / "split_tasks.json"
    if not split_path.exists():
        return
    try:
        splits = json.loads(split_path.read_text())
    except json.JSONDecodeError as e:
        r.fail(f"`split_tasks.json` tiene JSON inválido: {e}")
        return

    if "base" not in splits:
        r.fail('`split_tasks.json` debe contener el split "base" (obligatorio)')
    else:
        r.ok('`split_tasks.json` contiene el split "base"')

    # The spec explicitly requires train and test: "Incluir las divisiones, train y test ademas de la base"
    for split_name in ["train", "test"]:
        if split_name in splits:
            r.ok(f'`split_tasks.json` contiene el split "{split_name}"')
        else:
            r.fail(
                f'`split_tasks.json` no tiene split "{split_name}" — '
                "el enunciado requiere incluir splits train, test y base"
            )

    # Cross-check: all split IDs should appear in tasks.json
    tasks_path = REPO_ROOT / "data" / "tau2" / "domains" / domain / "tasks.json"
    if tasks_path.exists():
        try:
            tasks = json.loads(tasks_path.read_text())
            task_ids = {str(t.get("id")) for t in tasks if isinstance(t, dict)}
            for split_name, ids in splits.items():
                missing_ids = [i for i in ids if str(i) not in task_ids]
                if missing_ids:
                    r.fail(
                        f'Split "{split_name}" en `split_tasks.json` referencia '
                        f"IDs que no existen en `tasks.json`: {missing_ids}"
                    )
        except (json.JSONDecodeError, TypeError):
            pass  # tasks.json error already reported separately


def check_registry(domain: str, r: CheckResult):
    """Check registry.py has both register_domain and register_tasks calls for the domain."""
    registry_path = REPO_ROOT / "src" / "tau2" / "registry.py"
    if not registry_path.exists():
        r.fail("`src/tau2/registry.py` no encontrado")
        return
    content = registry_path.read_text()

    has_import = bool(re.search(rf"from tau2\.domains\.{re.escape(domain)}\.", content))
    # Use re.DOTALL so .* matches across newlines (calls may be multi-line)
    has_register_domain = bool(re.search(
        rf'register_domain\s*\([^)]*"{re.escape(domain)}"', content, re.DOTALL
    ))
    has_register_tasks = bool(re.search(
        rf'register_tasks\s*\([^)]*"{re.escape(domain)}"', content, re.DOTALL
    ))

    if has_import:
        r.ok(f"El módulo `{domain}` es importado en `registry.py`")
    else:
        r.fail(
            f"No se encontró `from tau2.domains.{domain}.environment import ...` en `registry.py`"
        )

    if has_register_domain:
        r.ok(f"`registry.register_domain(...)` llamado para `{domain}`")
    else:
        r.fail(
            f'No se encontró `registry.register_domain(..., "{domain}")` en `registry.py`'
        )

    if has_register_tasks:
        r.ok(f"`registry.register_tasks(...)` llamado para `{domain}`")
    else:
        r.fail(
            f'No se encontró `registry.register_tasks(..., "{domain}")` en `registry.py`'
        )


def check_tools_count(domain: str, r: CheckResult):
    """Count @is_tool decorated methods in tools.py."""
    tools_path = REPO_ROOT / "src" / "tau2" / "domains" / domain / "tools.py"
    if not tools_path.exists():
        return
    content = tools_path.read_text()
    # Count @is_tool(...) decorator occurrences — each one corresponds to one tool
    tool_defs = re.findall(r"@is_tool\s*\(", content)
    n = len(tool_defs)

    if n >= MIN_TOOLS:
        r.ok(f"`tools.py` tiene {n} herramientas decoradas con `@is_tool` (mínimo {MIN_TOOLS})")
    elif n > 0:
        r.fail(
            f"`tools.py` tiene solo {n} herramienta(s) decoradas con `@is_tool` — "
            f"se requieren al menos {MIN_TOOLS}"
        )
    else:
        # Fallback heuristic: count public methods if no @is_tool found
        public_methods = re.findall(r"^\s{4}def ([a-z]\w+)\(", content, re.MULTILINE)
        public_methods = [m for m in public_methods if not m.startswith("_")]
        if public_methods:
            r.warn(
                f"`tools.py` no usa el decorador `@is_tool` — "
                f"se encontraron {len(public_methods)} método(s) públicos, "
                "pero deben estar decorados con `@is_tool(ToolType.READ/WRITE/GENERIC)` "
                "para que el framework los registre como herramientas"
            )
        else:
            r.fail(
                f"`tools.py` no define herramientas con `@is_tool`. "
                f"Se requieren al menos {MIN_TOOLS}."
            )


def check_tests(domain: str, r: CheckResult):
    """Check test directory and test file exist.

    The spec says: tests/domain_tests/<domain>/
    The existing repo convention is: tests/test_domains/test_<domain>/
    We check the spec path first, then fall back to the repo convention.
    """
    spec_path = REPO_ROOT / "tests" / "domain_tests" / domain
    repo_path = REPO_ROOT / "tests" / "test_domains" / f"test_{domain}"

    if spec_path.exists():
        tests_base = spec_path
        display_path = f"tests/domain_tests/{domain}/"
    elif repo_path.exists():
        tests_base = repo_path
        display_path = f"tests/test_domains/test_{domain}/"
    else:
        r.fail(
            f"Falta directorio de tests — esperado en "
            f"`tests/domain_tests/{domain}/` "
            f"(o `tests/test_domains/test_{domain}/` siguiendo la convención del repo)"
        )
        return

    r.ok(f"Directorio de tests encontrado: `{display_path}`")

    init = tests_base / "__init__.py"
    if init.exists():
        r.ok(f"`__init__.py` presente en `{display_path}`")
    else:
        r.warn(f"Falta `__init__.py` en `{display_path}`")

    # Accept test_tools_<domain>.py or test_tools_<domain>_*.py
    test_files = list(tests_base.glob(f"test_tools_{domain}*.py"))
    if test_files:
        r.ok(f"Archivo de tests encontrado: `{test_files[0].name}`")
    else:
        any_tests = list(tests_base.glob("test_*.py"))
        if any_tests:
            r.warn(
                f"Tests encontrados pero sin el nombre requerido "
                f"`test_tools_{domain}.py`: {[f.name for f in any_tests]}"
            )
        else:
            r.fail(
                f"No se encontró `test_tools_{domain}.py` en `{display_path}` — "
                "debes incluir tests unitarios para cada herramienta"
            )


SIM_DIR = REPO_ROOT / "data" / "simulations"


def _find_simulation_file(domain: str) -> Path | None:
    """Return the most recent simulation file for the domain, or None."""
    if not SIM_DIR.exists():
        return None
    matches = sorted(SIM_DIR.glob(f"*{domain}*.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def check_simulation_file(domain: str, r: CheckResult):
    """Check that a simulation output file for the domain is present in data/simulations/."""
    if not SIM_DIR.exists():
        r.fail(
            f"No se encontró el directorio `data/simulations/` — "
            "debes ejecutar la simulación completa e incluir el archivo .json generado en tu PR "
            "(ver instrucciones-throttling-estudiantes.txt)"
        )
        return

    match = _find_simulation_file(domain)
    if match:
        r.ok(f"Archivo de simulación encontrado: `data/simulations/{match.name}`")
    else:
        all_sims = list(SIM_DIR.glob("*.json"))
        if all_sims:
            r.fail(
                f"No se encontró ningún archivo de simulación para el dominio `{domain}` "
                f"en `data/simulations/` — "
                f"el nombre del archivo debe contener el nombre del dominio. "
                f"Ejecuta: `tau2 run --domain {domain} ...` antes de enviar el PR."
            )
        else:
            r.fail(
                "No se encontró ningún archivo de simulación en `data/simulations/` — "
                "debes ejecutar la simulación completa e incluir el archivo .json generado en tu PR "
                "(ver instrucciones-throttling-estudiantes.txt)"
            )


def check_simulation_content(domain: str, r: CheckResult):
    """Validate that the simulation file is a complete run over all domain tasks.

    Checks:
    - The domain name inside the file matches this domain.
    - All task IDs from the 'base' split (or all tasks) are present in the simulation.
    - Every configured task has at least one completed simulation entry.
    - All simulation entries have reward_info (i.e., evaluation ran to completion).
    - Reports coverage percentage and a per-task breakdown of missing tasks.
    """
    sim_file = _find_simulation_file(domain)
    if sim_file is None:
        return  # already reported by check_simulation_file

    # --- Parse the simulation file ---
    try:
        data = json.loads(sim_file.read_text())
    except json.JSONDecodeError as e:
        r.fail(f"El archivo de simulación `{sim_file.name}` tiene JSON inválido: {e}")
        return

    # --- 1. Domain name matches ---
    sim_domain = data.get("info", {}).get("environment_info", {}).get("domain_name", "")
    if sim_domain != domain:
        r.fail(
            f"El archivo de simulación pertenece al dominio `{sim_domain}`, "
            f"no a `{domain}` — asegúrate de incluir el archivo de simulación correcto."
        )
        return  # rest of checks would be meaningless
    r.ok(f"El archivo de simulación corresponde al dominio `{domain}`")

    # --- 2. Load expected task IDs from the domain's tasks ---
    # Use 'base' split if available, otherwise all tasks
    expected_ids: set[str] = set()
    split_path = REPO_ROOT / "data" / "tau2" / "domains" / domain / "split_tasks.json"
    tasks_path = REPO_ROOT / "data" / "tau2" / "domains" / domain / "tasks.json"

    if split_path.exists():
        try:
            splits = json.loads(split_path.read_text())
            if "base" in splits:
                expected_ids = {str(i) for i in splits["base"]}
        except (json.JSONDecodeError, TypeError):
            pass

    if not expected_ids and tasks_path.exists():
        try:
            tasks = json.loads(tasks_path.read_text())
            expected_ids = {str(t["id"]) for t in tasks if isinstance(t, dict) and "id" in t}
        except (json.JSONDecodeError, TypeError):
            pass

    if not expected_ids:
        r.warn("No se pudo determinar el conjunto esperado de tareas para verificar la cobertura")
        return

    # --- 3. Check which tasks were configured in the simulation run ---
    sim_tasks = data.get("tasks", [])
    configured_ids = {str(t["id"]) for t in sim_tasks if isinstance(t, dict) and "id" in t}

    not_configured = expected_ids - configured_ids
    if not_configured:
        r.fail(
            f"La simulación no fue ejecutada sobre todas las tareas del split 'base' — "
            f"faltan {len(not_configured)}/{len(expected_ids)} tareas: "
            f"{sorted(not_configured)}. "
            f"Ejecuta `tau2 run --domain {domain}` sin filtrar tareas."
        )
    else:
        r.ok(
            f"La simulación incluyó las {len(expected_ids)} tareas del split 'base'"
        )

    # --- 4. Check which tasks have completed simulation entries ---
    simulations = data.get("simulations", [])
    completed_ids = {str(s["task_id"]) for s in simulations if isinstance(s, dict) and "task_id" in s}
    num_trials = data.get("info", {}).get("num_trials", 1)

    missing_completions = expected_ids - completed_ids
    coverage_pct = 100 * len(completed_ids & expected_ids) / len(expected_ids)

    if not missing_completions:
        r.ok(
            f"Cobertura completa: las {len(expected_ids)} tareas tienen simulación completada "
            f"({coverage_pct:.0f}%)"
        )
    else:
        r.fail(
            f"Cobertura incompleta: solo {len(completed_ids & expected_ids)}/{len(expected_ids)} "
            f"tareas tienen simulación ({coverage_pct:.0f}%) — "
            f"tareas sin completar: {sorted(missing_completions)}. "
            f"El run fue interrumpido o cancelado antes de terminar."
        )

    # --- 5. Check per-task trial count ---
    from collections import Counter
    trial_counts = Counter(str(s["task_id"]) for s in simulations if isinstance(s, dict))
    under_trial = [
        tid for tid in expected_ids
        if trial_counts.get(tid, 0) < num_trials
    ]
    if not under_trial:
        r.ok(
            f"Todas las tareas tienen {num_trials} trial(s) completado(s) "
            f"(num_trials={num_trials})"
        )
    else:
        r.fail(
            f"{len(under_trial)} tarea(s) tienen menos de {num_trials} trial(s): "
            f"{sorted(under_trial)}"
        )

    # --- 6. Check all simulation entries have reward_info ---
    missing_reward = [
        str(s.get("task_id", "?"))
        for s in simulations
        if isinstance(s, dict) and not s.get("reward_info")
    ]
    if not missing_reward:
        r.ok("Todas las simulaciones tienen `reward_info` (evaluación completada)")
    else:
        r.fail(
            f"{len(missing_reward)} simulación(es) no tienen `reward_info` — "
            "la evaluación no se completó correctamente para: "
            f"{sorted(set(missing_reward))}"
        )

    # --- 7. Informational: pass rate summary ---
    rewards = [
        s["reward_info"]["reward"]
        for s in simulations
        if isinstance(s, dict) and s.get("reward_info") and "reward" in s["reward_info"]
    ]
    if rewards:
        pass_rate = 100 * sum(1 for rw in rewards if rw >= 1.0) / len(rewards)
        avg_reward = sum(rewards) / len(rewards)
        r.ok(
            f"Resumen de resultados: pass rate={pass_rate:.0f}% "
            f"({sum(1 for rw in rewards if rw >= 1.0)}/{len(rewards)} tareas), "
            f"reward promedio={avg_reward:.2f}"
        )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_comment(domain: str, r: CheckResult) -> str:
    lines = []
    lines.append(f"## Revisión automática — Entrega 1 (`{domain}`)")
    lines.append("")

    total = len(r.passed) + len(r.warnings) + len(r.failed)
    score = len(r.passed)

    if r.success and not r.warnings:
        lines.append(
            f"> ✅ **Todo en orden.** Tu entrega cumple todos los requisitos estructurales "
            f"({score}/{total} verificaciones aprobadas)."
        )
    elif r.success:
        lines.append(
            f"> ⚠️ **Aprobado con advertencias.** La estructura es válida pero hay puntos a revisar "
            f"({score}/{total} verificaciones aprobadas)."
        )
    else:
        lines.append(
            f"> ❌ **Hay errores que corregir** antes de que el PR pueda ser aprobado "
            f"({score}/{total} verificaciones aprobadas)."
        )

    lines.append("")

    if r.passed:
        lines.append("### Verificaciones aprobadas")
        for msg in r.passed:
            lines.append(f"- ✅ {msg}")
        lines.append("")

    if r.warnings:
        lines.append("### Advertencias")
        for msg in r.warnings:
            lines.append(f"- ⚠️ {msg}")
        lines.append("")

    if r.failed:
        lines.append("### Errores a corregir")
        for msg in r.failed:
            lines.append(f"- ❌ {msg}")
        lines.append("")

    lines.append("---")
    lines.append("*Revisión automática — bot del curso IRN 2026*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--changed-files", nargs="+", required=True,
        help="List of files changed in the PR"
    )
    parser.add_argument(
        "--domain", default=None,
        help="Override domain name detection"
    )
    args = parser.parse_args()

    domain = args.domain or detect_domain(args.changed_files)

    if domain is None:
        print("## Revisión automática — Entrega 1")
        print("")
        print("> **No se pudo detectar el dominio del estudiante.**")
        print("> Asegúrate de que tus archivos estén en `src/tau2/domains/<nombre_dominio>/`.")
        print("> Si hay múltiples dominios en el PR, asegúrate de que solo hayas")
        print("> modificado archivos de tu propio dominio.")
        print("> Si crees que es un error, menciona al profesor en el PR.")
        sys.exit(1)

    r = CheckResult()

    # Run all checks
    check_python_module(domain, r)
    check_data_model_content(domain, r)
    check_environment_content(domain, r)
    check_data_files(domain, r)
    check_db_json(domain, r)
    check_tasks_json(domain, r)
    check_split_tasks(domain, r)
    check_registry(domain, r)
    check_tools_count(domain, r)
    check_tests(domain, r)
    check_simulation_file(domain, r)
    check_simulation_content(domain, r)

    comment = format_comment(domain, r)
    print(comment)

    # Write to file for GH Actions step to pick up
    output_path = Path("/tmp/pr_comment.md")
    output_path.write_text(comment)

    sys.exit(0 if r.success else 1)


if __name__ == "__main__":
    main()
