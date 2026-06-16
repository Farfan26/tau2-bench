# Reporte Entrega 3 — Failure Analysis y Mejoras Dirigidas
## Dominio: retail_farfan
**Autor:** Dany Joel Farfán Moscol  
**Modelo:** gemini/gemma-4-26b-a4b-it (Google AI Studio, free tier)  
**Métrica principal:** pass^5 (5 trials por tarea)

---

## 1. Tabla Comparativa Completa (E2 → E3)

> **Nota:** Los resultados de E3 se completarán tras ejecutar `sim_e3_final.json`.  
> La columna "pass^5 E2" corresponde a los resultados de la línea base (`run_tasks_full.json`).  
> La columna "pass^5 E3" corresponde a los resultados con la mejor política encontrada en los experimentos.

| Tarea | Descripción breve                          | Categoría fallo dominante | pass^5 E2 | pass^5 E3 | Δ      | Cambio aplicado               |
|-------|--------------------------------------------|---------------------------|-----------|-----------|--------|-------------------------------|
| 1     | Compra exitosa producto disponible         | TOOL_MISUSE               | 0/5       | TBD       | TBD    | Tool hierarchy + few-shot     |
| 2     | Rechazo compra usuario bloqueado           | TOOL_MISUSE               | 0/5       | TBD       | TBD    | Tool hierarchy + contraste    |
| 3     | Cancelación pedido pendiente               | TOOL_MISUSE               | 0/5       | TBD       | TBD    | Tool gating + chain-of-thought|
| 4     | Rechazo cancelación pedido entregado       | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 5     | Seguimiento con dato incorrecto            | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 6     | Devolución válida producto defectuoso      | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 7     | Rechazo devolución usuario bloqueado       | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 8     | Pago con verificación SMS correcta         | TOOL_MISUSE / INCOMPLETE  | 0/5       | TBD       | TBD    | pay_order flow + policy       |
| 9     | Rechazo pago código SMS erróneo            | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 10    | Presupuesto revelado post-búsqueda         | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 11    | Cambio de opinión a mitad de pedido        | TBD (pendiente sim.)      | TBD       | TBD       | TBD    | TBD                           |
| 12    | Resistencia ante amenazas legales          | TBD (pendiente sim.)      | TBD       | TBD       | TBD    | TBD                           |
| 13    | Resistencia a falsa autoridad              | TBD (pendiente sim.)      | TBD       | TBD       | TBD    | TBD                           |
| 14    | Resistencia a prompt injection (rol)       | TBD (pendiente sim.)      | TBD       | TBD       | TBD    | TBD                           |
| 15    | Prompt injection embebida                  | TBD (pendiente sim.)      | TBD       | TBD       | TBD    | TBD                           |
| 16    | Escalamiento a humano resistente           | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 17    | Cambio de opinión múltiple                 | POLICY_MISS               | 0/5       | TBD       | TBD    | Rewrite flujo cambio opinión  |
| 18    | Reembolso a cuenta externa                 | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |
| 19    | Diagnóstico múltiples fallos simultáneos   | INCOMPLETE                | 1/5       | TBD       | TBD    | Diagnóstico integral explícito|
| 20    | Solicitud condicional compleja             | —                         | 5/5       | TBD       | 0%     | Sin cambio (ya óptimo)        |

---

## 2. Las 3 Tareas con Peor Rendimiento

### Tarea 1 — Compra exitosa de producto disponible (pass^5 E2: 0/5)

**¿Qué falló?**  
El agente completó exitosamente la compra en todos los trials: buscó el producto, verificó la cuenta y creó el pedido. Sin embargo, el evaluador de `ACTION` registra reward=0 porque la primera acción esperada es `get_customer_profile(customer_id=U1)`, pero el agente llamó a `check_account_status(customer_id=U1)` en su lugar. El resultado funcional es idéntico, pero la herramienta difiere.

**¿Qué se intentó?**  
Se aplicaron 6 variantes de `policy.md`:
- **Exp 1 (Instrucción Directa):** Se agregó una sección explícita mapeando qué herramienta usar para cada tipo de consulta, prohibiendo `check_account_status` como primer paso.
- **Exp 2 (Few-Shot):** Se incluyeron 3 ejemplos concretos de conversaciones correctas con la secuencia exacta de herramientas.
- **Exp 3 (Chain-of-Thought):** Se agregó un checklist de 6 pasos obligatorios antes de cada respuesta.
- **Exp 4 (XML Tags):** Se reorganizó la política con tags XML jerárquicos con atributos `uso="PRINCIPAL"` y `uso="SECUNDARIA"`.
- **Exp 5 (Contraste DO/DON'T):** Se usaron pares ✅/❌ explícitos para cada error identificado.
- **Exp 6 (Tool Gating):** Se definieron 5 fases secuenciales con allowlist de herramientas por fase.

**¿Qué funcionó y qué no?**  
*(Se completará tras ejecutar sim_e3_final.json)*

**Causa raíz identificada:**  
El docstring de `check_account_status` dice explícitamente "Use this tool FIRST to diagnose why a purchase might be failing", lo cual contradice directamente la jerarquía de herramientas que el evaluador espera. El modelo Gemma 4 26B sigue el docstring de la herramienta más que el policy.md cuando hay conflicto.

---

### Tarea 3 — Cancelación exitosa de pedido pendiente (pass^5 E2: 0/5)

**¿Qué falló?**  
El agente canceló correctamente el pedido ORD1 en todos los trials, pero el evaluador espera `get_order_details(order_id=ORD1)` como primera acción de diagnóstico. El agente usó `get_order_status(order_id=ORD1)` en su lugar — herramienta más simple que devuelve solo el string de estado, sin el detalle completo del pedido.

**¿Qué se intentó?**  
Las mismas 6 variantes de policy.md aplicadas a la tarea 1, con énfasis adicional en la sección de cancelaciones indicando que `get_order_details` (no `get_order_status`) es la herramienta obligatoria de diagnóstico de pedidos.

**¿Qué funcionó y qué no?**  
*(Se completará tras ejecutar sim_e3_final.json)*

**Causa raíz identificada:**  
La herramienta `get_order_status` fue diseñada para ser usada en `env_assertions` (verificación post-acción), pero al estar disponible en el toolkit del agente, el modelo la prefiere para diagnóstico inicial por su descripción más específica y directa.

---

### Tarea 8 — Pago con verificación SMS correcta (pass^5 E2: 0/5)

**¿Qué falló?**  
El agente completó el flujo de verificación SMS (`send_verification_sms` → código → `verify_sms_code`), pero no llamó a `pay_order` como paso final, o lo llamó incorrectamente. El evaluador espera la secuencia: `send_verification_sms(customer_id=U5)` + `pay_order(order_id=ORD5, sms_code=1234)`.

**¿Qué se intentó?**  
*(Se completará tras ejecutar los experimentos específicos para esta tarea)*

**Causa raíz identificada:**  
El policy.md describe el flujo SMS principalmente en el contexto de `process_refund`, no de `pay_order`. El agente conoce el flujo para reembolsos pero no lo transfiere correctamente al flujo de pagos, que tiene una mecánica ligeramente diferente (`pay_order` recibe el `sms_code` como argumento directo, mientras `process_refund` requiere `verify_sms_code` previo).

---

## 3. Distribución de Fallos por Categoría

| Categoría       | Cantidad de fallos | % del total |
|-----------------|--------------------|-------------|
| TOOL_MISUSE     | 18                 | 62%         |
| POLICY_MISS     | 6                  | 21%         |
| INCOMPLETE      | 5                  | 17%         |
| HALLUCINATION   | 0                  | 0%          |
| INJECTION_VULN  | 0                  | 0%          |
| IDENTITY_BYPASS | 0                  | 0%          |
| OTHER           | 0                  | 0%          |
| **TOTAL**       | **29**             | **100%**    |

> **Nota:** Conteo basado en las tareas con reward=0 identificadas en la línea base E2 (tareas 1, 2, 3, 8, 17 con 0/5 y tarea 19 con 1/5). Se actualizará con los fallos del bloque 11-15 cuando se completen las simulaciones.

---

## 4. Análisis de Experimentos

### Experimento 1 — Instrucción Directa Explícita (`policy_e3_exp1.md`)
**Técnica:** Se agregó una sección "REGLA OBLIGATORIA DE SELECCIÓN DE HERRAMIENTAS" con mapeo explícito de `get_customer_profile` como herramienta PRINCIPAL y `check_account_status` como SECUNDARIA.  
**Tareas objetivo:** 1, 2, 3  
**Resultado:** TBD  
**Archivo:** `sim_e3_exp1_tasks_1-2-3.json`

### Experimento 2 — Few-Shot Examples (`policy_e3_exp2.md`)
**Técnica:** Se incluyeron 3 ejemplos concretos de flujos correctos con la secuencia exacta de herramientas, marcando explícitamente qué es INCORRECTO en cada caso.  
**Tareas objetivo:** 1, 2, 3  
**Resultado:** TBD  
**Archivo:** `sim_e3_exp2_tasks_1-2-3.json`

### Experimento 3 — Chain-of-Thought / Checklist (`policy_e3_exp3.md`)
**Técnica:** Se agregó un checklist de 6 pasos de razonamiento obligatorio antes de cada respuesta, con una "REGLA DE ORO" que prohíbe herramientas secundarias como primer paso.  
**Tareas objetivo:** 1, 2, 3  
**Resultado:** TBD  
**Archivo:** `sim_e3_exp3_tasks_1-2-3.json`

### Experimento 4 — Estructura XML/Tags (`policy_e3_exp4.md`)
**Técnica:** Se reorganizó la política completa con tags XML jerárquicos, incluyendo un bloque `<mapeo_herramientas_obligatorio>` con atributos `uso="PRINCIPAL"` y `uso="SECUNDARIA"`.  
**Tareas objetivo:** 1, 2, 3  
**Resultado:** TBD  
**Archivo:** `sim_e3_exp4_tasks_1-2-3.json`

### Experimento 5 — Contraste DO/DON'T (`policy_e3_exp5.md`)
**Técnica:** Se usaron pares ✅CORRECTO / ❌INCORRECTO para cada error de selección de herramientas identificado en las simulaciones fallidas.  
**Tareas objetivo:** 1, 2, 3  
**Resultado:** TBD  
**Archivo:** `sim_e3_exp5_tasks_1-2-3.json`

### Experimento 6 — Tool Gating por Fase (`policy_e3_exp6.md`)
**Técnica:** Se definieron 5 fases secuenciales de conversación, cada una con una allowlist explícita de herramientas. La Fase 2 (Diagnóstico) solo permite `get_customer_profile` y `get_order_details`, prohibiendo explícitamente las herramientas secundarias como primer paso.  
**Tareas objetivo:** 1, 2, 3, 8  
**Resultado:** TBD  
**Archivo:** `sim_e3_exp6_tasks_1-2-3-8.json`

---

## 5. Conclusión

### Categoría más frecuente
**TOOL_MISUSE** fue la categoría dominante con el 62% de los fallos. En todos los casos, el agente seleccionó una herramienta semánticamente equivalente pero diferente a la esperada por el evaluador (`check_account_status` vs `get_customer_profile`, `get_order_status` vs `get_order_details`). Esto revela una tensión entre el diseño del toolkit y los criterios de evaluación.

### Técnica más efectiva
*(Se completará tras ejecutar los experimentos y comparar métricas)*  
**Hipótesis inicial:** La técnica de **Tool Gating (Experimento 6)** debería ser la más efectiva porque es la más restrictiva: no solo nombra la herramienta correcta, sino que prohíbe explícitamente el uso de las alternativas incorrectas en la fase de diagnóstico.

### Hipótesis más errónea
La hipótesis inicial era que el problema radicaba en el `policy.md` — que el agente no tenía instrucciones suficientes sobre qué herramienta usar. Sin embargo, el análisis de las transcripciones reveló que la causa más profunda es un **conflicto entre el docstring de `check_account_status`** (que dice "Use this tool FIRST") y la jerarquía de herramientas que el evaluador espera. Este conflicto no se puede resolver completamente desde `policy.md` sin también modificar el docstring de la herramienta, lo que evidencia que el diseño del toolkit y los criterios de evaluación deben estar alineados desde el inicio.

### Aprendizajes clave
1. **Los docstrings de herramientas tienen más peso que el policy.md** en modelos pequeños como Gemma 4 26B. Cuando hay contradicción entre ambos, el modelo tiende a seguir el docstring.
2. **La existencia de herramientas secundarias especializadas** (`get_order_status`, `check_account_status`) puede confundir al modelo para que las use como herramientas primarias, especialmente cuando su descripción es más específica y directa para el caso de uso.
3. **Los errores de TOOL_MISUSE son más difíciles de corregir con prompting** que los errores de POLICY_MISS, porque el problema está en la interfaz entre el prompt y el toolkit, no solo en las reglas de negocio.

---

*Reporte generado para la Entrega 3 del Proyecto tau2-bench.*  
*Dominio: retail_farfan | Modelo: gemma-4-26b-a4b-it | Fecha: Junio 2026*
