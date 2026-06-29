# Guía: experimentos E4 con `--env-args` (dominio burger)

## --env-args

`--env-args` es un argumento de `tau2 run` que pasa un diccionario JSON directamente
a `get_environment()`. Permite cambiar la estrategia de chunking, el número de chunks
recuperados y el think tool **sin tocar ningún archivo Python** entre una condición y otra.

Para el dominio `burger` (y para cualquier dominio que implemente el patrón de E4),
los parámetros disponibles son:

| Parámetro          | Tipo   | Default     | Descripción |
|--------------------|--------|-------------|-------------|
| `chunking_strategy`| string | `"headers"` | Cómo dividir `policy.md`: `"headers"`, `"fixed_200"`, `"fixed_400"`, `"sentence_window"` |
| `retrieval_k`      | int    | `3`         | Número de chunks devueltos por cada llamada a `retrieve_policy` |
| `use_think`        | bool   | `false`     | Si `true`, añade automáticamente las instrucciones de `think()` al system prompt |
| `use_rag`          | bool   | `true`      | Si `false`, desactiva RAG completamente (útil para baseline) |

**Embeddings:** usa `gemini-embedding-001` vía `google-genai` — el mismo modelo del notebook
del curso. Requiere `GEMINI_API_KEY` o `GOOGLE_API_KEY` (ya configurada desde E3).
Los embeddings tienen su **propio límite de 100 RPM y 1 000 solicitudes/día** en el free tier,
separado del límite de chat. El framework aplica throttling automático a embeddings:
espera proactivamente antes de cada llamada si el límite de RPM está lleno, y lanza
`RuntimeError` si se agota el cupo diario.
Los tests usan `_fake_embed` y no consumen cuota.

---

## Estructura del comando base

Todos los comandos de E4 siguen esta estructura. Reemplaza `<TU-MODELO>` con el modelo
que usaste en E3 (ej. `gemini/gemma-4-26b-a4b-it`) y `<bucket>` con un nombre descriptivo
para el bucket de rate limiting (debe ser **el mismo** en agent y user para compartir el pool).

```bash
tau2 run \
  --domain burger \
  --agent-llm <TU-MODELO> \
  --user-llm  <TU-MODELO> \
  --task-ids burger_order_1 burger_order_2 burger_order_3 \
             burger_order_4 burger_order_5 burger_order_6 \
  --num-trials 5 \
  --max-steps 30 \
  --max-concurrency 1 \
  --save-to <nombre-archivo> \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14,
    "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000,
    "rate_limit_bucket": "<bucket>", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14,
    "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000,
    "rate_limit_bucket": "<bucket>", "rate_limit_token_reserve": 750}' \
  --env-args '<JSON>'
```

> Para tu dominio real: reemplaza los `--task-ids` con las 10 tareas de menor pass^5 en E3.

---

## Condiciones del experimento

### Condición A — Baseline sin RAG (reutiliza E3)

Esta condición **no requiere ejecutar nada nuevo**: copia el JSON de tu mejor simulación
de E3 como `sim_e4_A_baseline.json`. Si necesitas regenerarla:

```bash
tau2 run \
  --domain burger \
  --agent-llm gemini/gemma-4-26b-a4b-it \
  --user-llm  gemini/gemma-4-26b-a4b-it \
  --task-ids burger_order_1 burger_order_2 burger_order_3 \
             burger_order_4 burger_order_5 burger_order_6 \
  --num-trials 5 \
  --max-steps 30 \
  --max-concurrency 1 \
  --save-to sim_e4_A_baseline \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"use_rag": false}'
```

---

### Condición B — RAG con `headers`, k=3, sin think

```bash
tau2 run \
  --domain burger \
  --agent-llm gemini/gemma-4-26b-a4b-it \
  --user-llm  gemini/gemma-4-26b-a4b-it \
  --task-ids burger_order_1 burger_order_2 burger_order_3 \
             burger_order_4 burger_order_5 burger_order_6 \
  --num-trials 5 \
  --max-steps 30 \
  --max-concurrency 1 \
  --save-to sim_e4_B_headers_k3 \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"chunking_strategy": "headers", "retrieval_k": 3}'
```

---

### Condición C1 — RAG con `fixed_200`, k=3, sin think

```bash
tau2 run \
  --domain burger \
  --agent-llm gemini/gemma-4-26b-a4b-it \
  --user-llm  gemini/gemma-4-26b-a4b-it \
  --task-ids burger_order_1 burger_order_2 burger_order_3 \
             burger_order_4 burger_order_5 burger_order_6 \
  --num-trials 5 \
  --max-steps 30 \
  --max-concurrency 1 \
  --save-to sim_e4_C1_fixed200_k3 \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"chunking_strategy": "fixed_200", "retrieval_k": 3}'
```

---

### Condición C2 — RAG con `fixed_400`, k=3, sin think

```bash
tau2 run \
  --domain burger \
  --agent-llm gemini/gemma-4-26b-a4b-it \
  --user-llm  gemini/gemma-4-26b-a4b-it \
  --task-ids burger_order_1 burger_order_2 burger_order_3 \
             burger_order_4 burger_order_5 burger_order_6 \
  --num-trials 5 \
  --max-steps 30 \
  --max-concurrency 1 \
  --save-to sim_e4_C2_fixed400_k3 \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"chunking_strategy": "fixed_400", "retrieval_k": 3}'
```

> B, C1 y C2 son independientes entre sí — córrelas en paralelo si tienes varias
> terminales disponibles, o en secuencia si prefieres monitorear una por una.

---

### Condición D — Mejor estrategia de B/C + think activado

Primero compara el pass^5 de B, C1 y C2. Luego re-corre la ganadora añadiendo
`"use_think": true`. El framework inyecta las instrucciones de `think()` en el
system prompt automáticamente — no necesitas editar ningún archivo.

Ejemplo con `fixed_400` (cambia `"fixed_400"` por la estrategia ganadora y ajusta `--save-to`):

```bash
tau2 run \
  --domain burger \
  --agent-llm gemini/gemma-4-26b-a4b-it \
  --user-llm  gemini/gemma-4-26b-a4b-it \
  --task-ids burger_order_1 burger_order_2 burger_order_3 \
             burger_order_4 burger_order_5 burger_order_6 \
  --num-trials 5 \
  --max-steps 30 \
  --max-concurrency 1 \
  --save-to sim_e4_D_fixed400_k3_think \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"chunking_strategy": "fixed_400", "retrieval_k": 3, "use_think": true}'
```

Si ganó `fixed_200`, cambia `"fixed_400"` → `"fixed_200"` y `sim_e4_D_fixed400_k3_think` → `sim_e4_D_fixed200_k3_think`.
Si ganó `headers`, cambia `"fixed_400"` → `"headers"`.

---

### Verificación rápida (1 tarea, 1 trial)

Usa este comando para comprobar que la instalación funciona antes de lanzar la corrida completa.
Consume ~10–20 llamadas de chat y ~5–10 llamadas de embedding (1 init al comenzar +
1 búsqueda por cada llamada a `retrieve_policy` durante la simulación).

> Si ya corriste esta verificación antes, el archivo `sim_verify_fixed400_think.json`
> existe y `tau2 run` intentará **reanudar** esa corrida en vez de empezar una nueva.
> Cambia `--save-to` a un nombre distinto o elimina el archivo para empezar fresco.

```bash
tau2 run \
  --domain burger \
  --agent-llm gemini/gemma-4-26b-a4b-it \
  --user-llm  gemini/gemma-4-26b-a4b-it \
  --task-ids burger_order_1 \
  --num-trials 1 \
  --max-steps 10 \
  --max-concurrency 1 \
  --save-to sim_verify_fixed400_think \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"chunking_strategy": "fixed_400", "retrieval_k": 3, "use_think": true}'
```

---

## Notas sobre throttling y cuota

- `--max-steps 30`: cualquier simulación que llegue al límite se detiene automáticamente
  y cuenta como fallida. Protege la cuota diaria si el agente entra en bucle.
- `--max-concurrency 1`: una simulación a la vez. Junto con el rate limiter interno
  (`rate_limit_requests_per_minute: 14`) garantiza no superar el límite de Google AI Studio.
- `--num-trials 5`: 5 intentos por tarea para calcular pass^5. Con 6 tareas × 5 trials
  son **30 simulaciones** por condición. Cada simulación tiene ~10–15 turnos, lo que
  equivale a **~300–450 llamadas LLM** (agente + usuario) por condición. Planificar
  ~1–2 horas por condición en la API gratuita.
- **rate_limit_bucket**: agent y user deben compartir el **mismo nombre** de bucket para
  que el rate limiter los trate como una sola fuente y no superen 14 RPM en total.
- **Embeddings**: `gemini-embedding-001` tiene límite propio de **100 RPM y 1 000 req/día**
  en el free tier, separado del presupuesto de chat. El framework hace throttling automático:
  espera si el RPM está lleno y lanza error si se agota el cupo diario. Solo se llaman al
  iniciar cada simulación (indexación) y en cada turno de `retrieve_policy`.

---

## Ver y mover resultados

```bash
# Abrir el visor interactivo (navega con teclado; presiona q para salir)
tau2 view --file data/simulations/sim_e4_B_headers_k3.json

# Mover al directorio del dominio una vez revisado
mv data/simulations/sim_e4_B_headers_k3.json \
   data/tau2/domains/burger/simulations/sim_e4_B_headers_k3.json
```

> `tau2 view` lanza un **visor interactivo**, no imprime directamente en terminal.
> Usa las flechas para navegar entre simulaciones y `q` para salir.

Hacer lo mismo para C1, C2 y D una vez completadas.
