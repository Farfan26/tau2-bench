# Evidencia de errores 500 con Gemma 4

Dominio: `hotel_calle`

Fecha de ejecucion: 2026-05-23

## Comando usado

Se ejecuto el dominio con Gemma 4 siguiendo el formato indicado por el profesor, cambiando el dominio a `hotel_calle` y filtrando tarea por tarea con `--task-ids` para poder guardar la evidencia de cada intento:

```bash
tau2 run \
  --domain hotel_calle \
  --agent-llm gemini/gemma-4-31b-it \
  --user-llm gemini/gemma-4-26b-a4b-it \
  --num-trials 1 \
  --max-concurrency 1 \
  --task-ids <task_id> \
  --save-to <archivo_de_salida> \
  --agent-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "google-free-tier-31b", "rate_limit_token_reserve": 750}' \
  --user-llm-args '{"temperature": 0.0, "rate_limit_requests_per_minute": 14, "rate_limit_requests_per_day": 14000, "rate_limit_tokens_per_minute": 150000, "rate_limit_bucket": "google-free-tier-26b", "rate_limit_token_reserve": 750}'
```

En Windows/PowerShell se agrego configuracion UTF-8 porque el CLI fallaba al imprimir simbolos como `✅`:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## Resultado observado

La Corrida 2 empezo correctamente:

| Tarea | Task ID | Estado | Reward |
|---:|---|---|---:|
| 1 | `hotel_availability_suite_may` | completa | 1.0 |
| 2 | `hotel_price_family_april` | completa | 0.0 |
| 3 | `hotel_cheapest_room_for_two` | completa | 1.0 |
| 4 | `hotel_reservation_double_success` | completa | 1.0 |
| 5 | `hotel_booking_ambiguous_missing_dates` | error API 500 | sin simulacion valida |
| 6 | `hotel_booking_ambiguous_missing_guest_details` | error API 500 | sin simulacion valida |

## Error visto en consola

Durante la ejecucion de las tareas 5 y 6, la API devolvio:

```text
litellm.InternalServerError: GeminiException InternalServerError
{
  "error": {
    "code": 500,
    "message": "Internal error encountered.",
    "status": "INTERNAL"
  }
}
```

Tambien se observo este endpoint en el traceback:

```text
https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent
```

## Archivos generados como evidencia

Estos archivos se generaron, pero quedaron sin simulaciones validas (`simulations=0`) debido al error 500:

| Archivo | Resultado |
|---|---|
| `data/simulations/hotel_calle_round_02_cli_task05_utf8_attempt01.json` | `simulations=0` |
| `data/simulations/hotel_calle_round_02_cli_task05_utf8_attempt02.json` | `simulations=0` |
| `data/simulations/hotel_calle_round_02_cli_task06_utf8_attempt01.json` | `simulations=0` |

Manifest parcial:

```text
data/tau2/domains/hotel_calle/simulations_manifest_round_02.csv
```

## Interpretacion

El dominio carga y ejecuta tareas correctamente, como se observa en las tareas 1 a 4 de la Corrida 2. El bloqueo aparece por errores externos de la API/modelo Gemma 4 durante llamadas posteriores, lo que impide completar todas las corridas necesarias para calcular `pass^10`.
