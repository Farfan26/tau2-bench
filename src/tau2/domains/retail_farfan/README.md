# 🛒 Dominio: Retail Farfán (E-commerce Support)

Este dominio simula un ecosistema de atención al cliente para una tienda minorista, diseñado para evaluar la capacidad de un agente de IA en la gestión de pedidos, devoluciones y soporte bilingüe dentro del framework **Tau2-Bench**.

## 📌 Descripción General
**Retail Farfán** permite interactuar con una base de datos dinámica. El sistema está optimizado para detectar el idioma del usuario automáticamente y aplicar reglas de negocio consistentes en **Español e Inglés**.

---

## 🏗️ Arquitectura del Dominio

### 1. Entidades Principales (`data_model.py`)
- **User (Usuario):** Datos del cliente y estado de cuenta (`activo`, `bloqueado`).
- **Product (Producto):** Artículos con precio, stock y políticas de devolución.
- **Order (Pedido):** Registro con estados: `pendiente`, `enviado`, `entregado`, `cancelado`.

### 2. Herramientas del Agente (`tools.py`)
El agente utiliza las siguientes funciones lógicas para resolver tareas:
- `get_user_details`: Valida la identidad y estado del cliente.
- `search_products`: Búsqueda robusta por palabra clave o **ID exacto** (optimizada para evitar errores de coincidencia).
- `create_order`: Crea pedidos validando stock y reglas de usuario.
- `track_order`: Consulta en tiempo real el estado logístico.
- `request_return`: Gestiona devoluciones basadas en el estado del pedido y tipo de producto.

---

### 🛡️ Políticas de Negocio (`policy.md`)
- **Validación:** No se entrega información sensible sin un `user_id` válido.
- **Cancelaciones:** Restringidas una vez que el estado es `shipped` (enviado).
- **Devoluciones:** Solo permitidas para pedidos `delivered` (entregados) y productos que lo permitan.
- **Multilingüismo:** El agente debe mantener el tono profesional y empático tanto en inglés como en español.

---

## 📊 Métricas de Rendimiento (Hito 1)
El dominio ha sido validado satisfactoriamente con el modelo **Gemini 1.5**:

| Métrica | Resultado |
| :--- | :--- |
| **Total de Tareas** | 20 casos de uso (Tasks) |
| **Average Reward** | **1.0000 (100% Éxito)** |
| **Consistency Check** | 100% (DB Match: YES) |

> **Nota sobre GitHub Actions:** El reporte de CI muestra una falla en el paso "Post-validación" debido a un error de permisos (`HttpError: Resource not accessible`). Se confirma que los tests de código (**Check tools, data_model, tasks**) pasaron satisfactoriamente y la lógica es funcional al 100%.

---

## 🚀 Cómo ejecutar la simulación
Para replicar los resultados localmente:

1. Configurar la API Key en `.env`:
   ```text
   GOOGLE_API_KEY=tu_clave_aqui

2. Ejecutar el benchmark:

Bash
python -m tau2.cli run --domain retail_farfan --num-trials 1


Autor: Dany Joel Farfán Moscol

Institución: Universidad de Piura (UDEP)

Carrera: Ingeniería Industrial y de Sistemas