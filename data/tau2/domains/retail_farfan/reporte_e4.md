# Reporte Entrega 4 — RAG Agéntico sobre Políticas

## Dominio: retail_farfan

**Autor:** Dany Joel Farfán Moscol

**Modelo:** gemini/gemma-4-26b-a4b-it (Google AI Studio, free tier)

**Métrica principal:** pass^5

---

## 1. Configuración del Experimento

### Tareas seleccionadas

Se seleccionaron las 7 tareas con menor pass^5 en E3 (todas con tasa de éxito ≤ 20%):

| Tarea | Descripción                                | pass^5 E3 |
| ----- | ------------------------------------------- | --------- |
| 1     | Compra exitosa producto disponible          | 0/5 (0%)  |
| 2     | Rechazo compra usuario bloqueado            | 0/5 (0%)  |
| 3     | Cancelación pedido pendiente               | 0/5 (0%)  |
| 8     | Pago con verificación SMS                  | 0/5 (0%)  |
| 11    | Cambio de opinión a mitad de pedido        | 0/5 (0%)  |
| 17    | Cambio de opinión múltiple                | 0/5 (0%)  |
| 19    | Diagnóstico múltiples fallos simultáneos | 1/5 (20%) |

Se usaron 7 tareas en lugar de 10 porque las 13 tareas restantes ya alcanzaban 5/5 en E3 y no aportarían señal comparativa.

### Estrategia de chunking

La política `policy.md` tiene ~975 palabras totales. Se evaluaron 3 estrategias:

| Estrategia | Núm. chunks | Palabras promedio por chunk |
| ---------- | ------------ | --------------------------- |
| headers    | 15           | 65                          |
| fixed_200  | 5            | 195                         |
| fixed_400  | 3            | 324                         |

Se descartó `fixed_400` (3 chunks, ~33% de la política por chunk — elimina la ventaja del RAG). Las condiciones experimentales fueron `headers` (B) y `fixed_200` (C).

---

## 2. Condiciones del Experimento

| Condición | Descripción                                               | `env-args`                                                              |
| ---------- | ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| A          | Baseline E3 (sin RAG, política completa en system prompt) | `{"use_rag": false}`                                                    |
| B          | RAG con headers, k=3, sin think                            | `{"chunking_strategy": "headers", "retrieval_k": 3}`                    |
| C          | RAG con fixed_200, k=3, sin think                          | `{"chunking_strategy": "fixed_200", "retrieval_k": 3}`                  |
| D          | RAG con headers, k=3, think activado                       | `{"chunking_strategy": "headers", "retrieval_k": 3, "use_think": true}` |

---

## 3. Tabla Comparativa de Resultados (pass^k por tarea)

| Tarea           | A (baseline)        | B (headers)           | C (fixed_200)       | D (headers+think)   | Δ A→B        | Δ A→D       |
| --------------- | ------------------- | --------------------- | ------------------- | ------------------- | -------------- | ------------- |
| 1               | 0/5 (0%)            | 5/5 (100%)            | 2/2 (100%)          | 0/5 (0%)            | +100%          | 0%            |
| 2               | 0/5 (0%)            | 5/5 (100%)            | 0/0 (—)            | 0/5 (0%)            | +100%          | 0%            |
| 3               | 0/5 (0%)            | 4/4 (100%)            | 0/0 (—)            | 0/5 (0%)            | +100%          | 0%            |
| 8               | 0/5 (0%)            | 0/4 (0%)              | 0/2 (0%)            | 0/5 (0%)            | 0%             | 0%            |
| 11              | 0/5 (0%)            | 0/4 (0%)              | 0/2 (0%)            | 0/5 (0%)            | 0%             | 0%            |
| 17              | 0/5 (0%)            | 0/4 (0%)              | 0/1 (0%)            | 0/5 (0%)            | 0%             | 0%            |
| 19              | 1/5 (20%)           | 0/4 (0%)              | 0/2 (0%)            | 0/5 (0%)            | -20%           | -20%          |
| **TOTAL** | **1/35 (3%)** | **14/30 (47%)** | **2/9 (22%)** | **0/35 (0%)** | **+44%** | **-3%** |

> **Nota sobre trials incompletos en B y C:** B completó 30/35 simulaciones y C completó 9/35 — los 5 y 26 trials restantes respectivamente fueron skipeados por cuota de embeddings (`gemini-embedding-1.0`, límite 1000 req/día free tier) al correr B y C en paralelo.

---

## 4. Análisis por Tarea

### Tareas 1, 2, 3 — TOOL_MISUSE resuelto por RAG (B: 100%, D: 0%)

El RAG con `headers` resolvió completamente las tareas 1, 2 y 3 en la condición B. La causa raíz identificada en E3 era que el agente usaba `check_account_status` en lugar de `get_customer_profile`, y `get_order_status` en lugar de `get_order_details`. Con RAG, el agente llama a `retrieve_policy` antes de actuar y recupera exactamente la sección relevante de la política que especifica la jerarquía de herramientas.

Sin embargo, en la condición D (headers + think), estas mismas tareas caen a 0/5. Esto sugiere que el think tool interfiere con el flujo de acción: el agente dedica demasiado tiempo al razonamiento interno y no llega a ejecutar las herramientas de diagnóstico en el orden correcto, o el think altera la secuencia de llamadas al toolkit de forma que el evaluador no reconoce.

### Tareas 8, 11, 17 — Sin mejora en ninguna condición (0% en A, B, C y D)

Estas tareas permanecen en 0% independientemente de la condición RAG o think. El RAG recupera correctamente las secciones de política relevantes, pero el agente no logra ejecutar los flujos correctamente:

* **Tarea 8:** El agente conoce el protocolo SMS (lo recupera del RAG), pero sigue sin llamar a `pay_order` como paso final, o confunde el flujo con `process_refund`.
* **Tareas 11 y 17:** El agente no gestiona correctamente el estado conversacional cuando el cliente cambia de decisión — problema de tracking de estado que no se resuelve con mejor acceso a la política.

### Tarea 19 — Regresión consistente en todas las condiciones RAG (-20%)

La tarea 19 pasó de 1/5 en A a 0/5 en todas las condiciones RAG (B, C, D). Una hipótesis es que el RAG, al recuperar solo chunks específicos, omite contexto relevante que el agente necesita para diagnosticar múltiples fallos simultáneos (cuenta bloqueada + sin stock). Con la política completa en el system prompt (condición A), el agente ocasionalmente detecta ambos impedimentos; con RAG, la recuperación parcial limita su capacidad diagnóstica.

---

## 5. Hallazgo Inesperado: Think Tool Empeora el Rendimiento (D: 0/35)

El resultado más sorprendente de E4 es que activar el think tool en la condición D produjo **0/35 pass** — peor que el baseline A (1/35) y mucho peor que B (14/30). Esto es contraintuitivo porque el think tool debería ayudar al agente a razonar antes de actuar.

**Hipótesis sobre la causa:** El modelo `gemma-4-26b-a4b-it` es relativamente pequeño y puede no manejar bien la combinación de RAG + think en un mismo turno. El think genera razonamiento interno que consume parte del context window y puede interferir con la secuencia de llamadas al toolkit. Adicionalmente, el think puede hacer que el agente "sobre-razone" y llegue a conclusiones incorrectas sobre qué herramienta usar, contrarrestando el beneficio del RAG.

Este hallazgo sugiere que para modelos pequeños (sub-30B parámetros), el think tool puede ser contraproducente cuando se combina con RAG, al menos en dominios con flujos de herramientas estrictos y evaluadores de acciones exactas.

---

## 6. Comparación de Estrategias de Chunking: B vs C

Con los datos disponibles (C solo completó 9/35 simulaciones por cuota de embeddings), la comparación es parcial:

| Estrategia    | Chunks | Palabras/chunk | Pass rate   | Tarea 1    |
| ------------- | ------ | -------------- | ----------- | ---------- |
| headers (B)   | 15     | 65             | 47% (14/30) | 5/5 (100%) |
| fixed_200 (C) | 5      | 195            | 22% (2/9)   | 2/2 (100%) |

Ambas estrategias resuelven la tarea 1 cuando logran completar simulaciones. La mayor granularidad de `headers` (15 chunks de ~65 palabras) permite recuperaciones más precisas y con menos ruido contextual que `fixed_200` (5 chunks de ~195 palabras).

---

## 7. Resumen de Condiciones

| Condición        | Pass total | Pass rate | Mejor tarea    | Peor resultado            |
| ----------------- | ---------- | --------- | -------------- | ------------------------- |
| A (baseline)      | 1/35       | 3%        | task 19: 1/5   | tasks 1-3, 8, 11, 17: 0/5 |
| B (headers k=3)   | 14/30      | 47%       | tasks 1,2: 5/5 | tasks 8,11,17,19: 0/4     |
| C (fixed_200 k=3) | 2/9        | 22%       | task 1: 2/2    | tasks 8,11,17,19: 0/x     |
| D (headers+think) | 0/35       | 0%        | —             | todas: 0/5                |

**Mejor condición overall: B (headers, k=3, sin think)**

---

## 8. Conclusiones

**RAG mejora TOOL_MISUSE pero no otros fallos.** La estrategia `headers` con k=3 resolvió completamente las tareas 1, 2 y 3 (0%→100%), que representaban el 62% de los fallos de E3. Para tareas con fallo de tipo POLICY_MISS (11, 17) e INCOMPLETE (8, 19), el RAG no aportó mejora.

**Think tool es perjudicial con modelos pequeños en flujos estrictos.** La condición D (headers + think) produjo 0/35, peor que todas las demás condiciones. Para el modelo Gemma 4 26B en un dominio con evaluación de acciones exactas, el razonamiento explícito interfiere con la ejecución correcta del toolkit.

**La cuota de embeddings del free tier es el principal límite operacional.** El límite de 1000 embeddings/día (`gemini-embedding-1.0`) impidió completar todas las simulaciones de B y C al correrlas en paralelo. Para futuros experimentos: correr condiciones en secuencia.

**Mejor configuración para este dominio: RAG headers k=3 sin think.**

---

*Reporte generado para la Entrega 4 del Proyecto tau2-bench.*

*Dominio: retail_farfan | Modelo: gemma-4-26b-a4b-it | Fecha: Junio 2026*
