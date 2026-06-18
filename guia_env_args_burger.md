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
Los embeddings tienen su **propio límite de 1 500 RPM**, separado del límite de chat.
Los tests usan `_fake_embed` y no consumen cuota.

---

## Estructura del comando base

Todos los comandos de E4 siguen esta estructura. Reemplaza `<TU-MODELO>` con el modelo
que usaste en E3 (ej. `gemini/gemma-4-27b-it`) y `<bucket>` con un nombre descriptivo
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

Ejemplo asumiendo que `fixed_200` ganó:

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
  --save-to sim_e4_D_fixed200_k3_think \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --user-llm-args  '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "gemma4-free-tier", "rate_limit_token_reserve": 750}' \
  --env-args '{"chunking_strategy": "fixed_200", "retrieval_k": 3, "use_think": true}'
```

Si en cambio ganó `headers`, cambia `"fixed_200"` → `"headers"` y ajusta `--save-to`.

---

## Notas sobre throttling y cuota

- `--max-steps 30`: cualquier simulación que llegue al límite se detiene automáticamente
  y cuenta como fallida. Protege la cuota diaria si el agente entra en bucle.
- `--max-concurrency 1`: una simulación a la vez. Junto con el rate limiter interno
  (`rate_limit_requests_per_minute: 14`) garantiza no superar el límite de Google AI Studio.
- `--num-trials 5`: 5 intentos por tarea para calcular pass^5. Con 6 tareas × 5 trials
  son **30 llamadas de agente + 30 de usuario simulador** mínimo por condición (sin contar
  turnos internos). Planificar ~1-2 horas por condición en la API gratuita.
- **rate_limit_bucket**: agent y user deben compartir el **mismo nombre** de bucket para
  que el rate limiter los trate como una sola fuente y no superen 14 RPM en total.
- **Embeddings**: `gemini-embedding-001` tiene límite propio de 1 500 RPM y no compite
  con el presupuesto de chat. Solo se llaman al iniciar cada simulación (indexación) y en
  cada turno de `retrieve_policy`.

---

## Ver y mover resultados

```bash
# Ver resultados en terminal
tau2 view --file data/simulations/sim_e4_B_headers_k3.json

# Mover al directorio del dominio
mv data/simulations/sim_e4_B_headers_k3.json \
   data/tau2/domains/burger/simulations/sim_e4_B_headers_k3.json
```

Hacer lo mismo para C1, C2 y D una vez completadas.
