# Hallazgos E1–E4: Síntesis del Curso

## Dominio: retail_farfan — Agente de Atención al Cliente

**Autor:** Dany Joel Farfán Moscol

**Modelo:** gemini/gemma-4-26b-a4b-it (Google AI Studio, free tier)

**Período:** Abril – Junio 2026

---

## 1. Descripción del Dominio

El dominio `retail_farfan` simula un agente de atención al cliente para una tienda de retail peruana. El agente maneja compras de productos, cancelaciones, devoluciones, reembolsos y pagos, con reglas de negocio que incluyen verificación de identidad por SMS (2FA), manejo de cuentas bloqueadas, y resistencia a manipulación (falsa autoridad, presión emocional, prompt injection).

El dominio fue diseñado con 20 tareas que cubren escenarios desde compras simples hasta diagnósticos multi-fallo y ataques adversariales. La base de datos incluye 5 usuarios (2 bloqueados), 4 productos (1 sin stock), y 5 pedidos con distintos estados.

---

## 2. Evolución del Agente a lo largo del Curso

### Entrega 1 — Construcción del dominio base

En E1 se construyó la infraestructura completa del dominio: `data_model.py` con 9 modelos Pydantic, `tools.py` con 20 herramientas decoradas con `@is_tool`, `environment.py`, `db.json`, `tasks.json` (20 tareas) y `split_tasks.json`. Una decisión de diseño crítica fue usar IDs determinísticos (SHA-256) en lugar de `uuid4()` para que el evaluador pudiera reproducir las simulaciones.

### Entrega 2 — Refinamiento de política y primeras simulaciones

En E2 se refinó `policy.md` con 6 variantes de prompting (instrucción directa, few-shot, chain-of-thought, XML, contraste DO/DON'T, tool gating). La línea base mostró **pass rate del 64%** (66/100 trials exitosos sobre las 20 tareas con 5 trials cada una). Las tareas con rendimiento perfecto (5/5) fueron la mayoría: tareas de resistencia adversarial (4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 18, 20) — el agente maneja bien las reglas de seguridad cuando tiene la política completa disponible.

### Entrega 3 — Diagnóstico de fallos y mejoras dirigidas

E3 identificó **3 categorías de fallo dominantes** mediante análisis manual de transcripciones JSON:

| Categoría  | %   | Tareas afectadas |
| ----------- | --- | ---------------- |
| TOOL_MISUSE | 62% | 1, 2, 3, 8       |
| POLICY_MISS | 21% | 17, 11           |
| INCOMPLETE  | 17% | 8, 19            |

El hallazgo más importante fue que el TOOL_MISUSE no era causado por una política incorrecta, sino por un **conflicto entre el docstring de `check_account_status`** (que decía "Use this tool FIRST") y la herramienta que el evaluador esperaba (`get_customer_profile`). Este tipo de conflicto entre el toolkit y el evaluador no se puede resolver solo desde `policy.md`.

### Entrega 4 — RAG Agéntico

E4 introduce RAG para que el agente consulte la política bajo demanda en lugar de recibirla completa en el system prompt. Los resultados muestran una mejora dramática en las tareas de TOOL_MISUSE y confirman que RAG no ayuda con otros tipos de fallo.

---

## 3. Evolución del Pass Rate a lo largo del Curso

| Entrega         | Condición                          | Tareas     | Trials | Pass rate    |
| --------------- | ----------------------------------- | ---------- | ------ | ------------ |
| E2 (baseline)   | Política completa en system prompt | 20         | 100    | 64% (64/100) |
| E3 (baseline)   | Mejor política de E2               | 20         | 100    | 66% (66/100) |
| E4-A (baseline) | Política completa, sin RAG         | 7 (peores) | 35     | 3% (1/35)    |
| E4-B            | RAG headers k=3                     | 7 (peores) | 30     | 47% (14/30)  |
| E4-C            | RAG fixed_200 k=3                   | 7 (peores) | 9      | 22% (2/9)    |
| E4-D            | RAG headers + think                 | 7 (peores) | 35     | 0% (0/35)    |

**Nota sobre E4-A:** El 3% en las 7 tareas más difíciles es consistente con E3 (esas mismas tareas tenían 0-20% en E3). El contraste A→B (+44%) muestra el impacto del RAG sobre esas tareas específicas.

---

## 4. Hallazgos Principales

### Hallazgo 1: El modelo Gemma 4 26B sigue los docstrings más que el policy.md

Durante E2 y E3, todas las variantes de prompting (instrucción directa, few-shot, chain-of-thought, XML, contraste, tool gating) fallaron en resolver TOOL_MISUSE para las tareas 1, 2 y 3. La causa no era la política — era el docstring de `check_account_status` que decía explícitamente "Use this tool FIRST". Cuando hay contradicción entre el docstring de una herramienta y el system prompt, el modelo pequeño (Gemma 4 26B) prioriza el docstring.

**Implicación:** Para modelos pequeños en producción, los docstrings de las herramientas son parte del "prompt efectivo" y deben estar alineados con los criterios de evaluación desde el diseño inicial.

### Hallazgo 2: RAG resuelve TOOL_MISUSE pero no otros tipos de fallo

El RAG con estrategia `headers` logró 100% en las tareas 1, 2 y 3 (de 0% en el baseline). Pero mantuvo 0% en las tareas 8, 11 y 17. Esto indica que:

* TOOL_MISUSE se origina en **falta de acceso contextual a la política correcta** — el agente no sabe qué herramienta usar porque el context window grande con la política completa diluye las instrucciones específicas. RAG lo resuelve al forzar una consulta explícita y enfocada.
* POLICY_MISS e INCOMPLETE se originan en **limitaciones del modelo para seguir flujos multi-paso** o **gestionar estado conversacional** — problemas que no se resuelven con mejor acceso a la política.

### Hallazgo 3: El límite de embeddings del free tier es el principal obstáculo operacional para RAG

El límite de 1000 embeddings/día (`gemini-embedding-1.0`, free tier) fue el cuello de botella principal en E4. Cada turno donde el agente llama a `retrieve_policy` consume un embedding. Con 7 tareas × 5 trials × ~5-10 turnos por conversación = 175-350 embeddings por condición, correr dos condiciones en paralelo agotó la cuota diaria antes de completar B y antes de poder lanzar D.

Para futuros proyectos con RAG en el free tier: correr condiciones en secuencia, no en paralelo, y estimar ~200-400 embeddings por condición de 35 simulaciones.

### Hallazgo 4: La estrategia de chunking headers es superior para políticas estructuradas por secciones

Para una política de ~975 palabras organizada en 13 secciones temáticas (como la de `retail_farfan`), la estrategia `headers` genera 15 chunks de ~65 palabras promedio. Esto produce chunks semánticamente cohesivos que mapean directamente a categorías de acción del agente ("Account Verification", "Orders & Cancellations", etc.).

En contraste, `fixed_200` genera 5 chunks de ~195 palabras que mezclan múltiples secciones, reduciendo la precisión de la recuperación. Para políticas bien estructuradas con `##` headers, la estrategia `headers` es la opción natural.

### Hallazgo 5: Los errores transitorios de API son inevitables en el free tier

A lo largo del curso, los errores más frecuentes fueron: 429 (rate limit), 500 (internal server error de Google), 403 (project blocked) y timeout. La actualización de `run.py` en E4 (que skipea trials fallidos en lugar de abortar toda la corrida) fue un cambio operacionalmente crítico — en E3, un solo error 500 abortaba las 35 simulaciones de un bloque; en E4, solo se pierde el trial específico.

---

## 5. Tabla Resumen: Categorías de Fallo y Técnicas que Funcionaron

| Categoría                 | Técnica que NO funcionó  | Técnica que SÍ funcionó | Tareas                    |
| -------------------------- | -------------------------- | -------------------------- | ------------------------- |
| TOOL_MISUSE (62%)          | Prompting (6 variantes E3) | RAG headers k=3 (E4-B)     | 1, 2, 3                   |
| TOOL_MISUSE flujo complejo | RAG headers k=3            | TBD (D con think)          | 8                         |
| POLICY_MISS                | Prompting, RAG             | TBD                        | 11, 17                    |
| INCOMPLETE                 | Prompting, RAG             | TBD                        | 19                        |
| Resistencia adversarial    | — (ya funciona 5/5)       | Política estándar        | 4-7, 9, 10, 12-16, 18, 20 |

---

## 6. Reflexiones Personales sobre el Proceso

A lo largo del curso aprendí que construir un agente de atención al cliente evaluable automáticamente es mucho más complejo de lo que parece. Los problemas que enfrenté no eran principalmente de código — eran de alineación entre tres componentes que deben estar coordinados: el toolkit (herramientas y sus docstrings), la política (policy.md), y el evaluador (tasks.json y sus criterios de reward).

El descubrimiento más sorprendente fue que seis técnicas diferentes de prompt engineering (E3) fallaron en resolver un problema que el RAG resolvió inmediatamente en E4. Esto me enseñó que a veces el problema no está donde parece: las tareas 1, 2 y 3 fallaban no porque la política fuera mala, sino porque el modelo nunca llegaba a leer la sección relevante cuando tenía toda la política disponible de golpe.

Los límites de la API gratuita fueron frustrantes pero también instructivos: me forzaron a planificar mejor el uso de recursos, a entender los distintos tipos de cuota (chat vs embeddings), y a ser más cuidadoso con las corridas en paralelo.

---

*Documento generado para la Entrega 4 del Proyecto tau2-bench.*

*Dominio: retail_farfan | Modelo: gemma-4-26b-a4b-it | Fecha: Junio 2026*
