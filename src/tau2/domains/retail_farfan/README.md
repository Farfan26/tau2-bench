El dominio **Retail Farfán** simula un sistema de gestión de pedidos para una tienda minorista. Permite a los usuarios consultar el estado de sus compras, verificar disponibilidad de productos y obtener detalles sobre la logística de entrega. Está diseñado para ser **completamente bilingüe (Español/Inglés)**, detectando el idioma del usuario automáticamente.

### 🏗️ Entidades Principales
- **Order (Pedido):** Incluye ID, lista de productos, estado (processing, shipped, delivered) y fecha.
- **Product (Producto):** Artículos con nombre, categoría y precio.
- **User (Usuario):** Datos del cliente para validación de identidad.

### 📋 Tareas Soportadas (Tasks)
1. **Consulta de Estado:** El usuario pregunta por el progreso de su pedido mediante un ID.
2. **Información de Envío:** El agente explica si un pedido puede ser cancelado o modificado según su estado.
3. **Soporte Bilingüe:** Resolución de dudas tanto en inglés como en español siguiendo las mismas reglas de negocio.

### 🛡️ Resumen de la Política (Policy)
- **Privacidad:** Solo se da información tras validar un ID de pedido existente.
- **Restricciones:** No se permiten cancelaciones si el pedido ya fue enviado (`shipped`).
- **Tono:** El agente debe ser empático y profesional en ambos idiomas.

### 📊 Métricas de Rendimiento
*(Adjunto pantallazo de la ejecución con las métricas obtenidas mediante el comando `tau2 run`).*