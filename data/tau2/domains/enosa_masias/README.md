# Dominio ENOSA - Asistente de Atención al Cliente

## Autor
Martin Masias

## Resumen del Dominio
El dominio `enosa_masias` simula un asistente virtual para la empresa de electricidad ENOSA. Ayuda a los clientes a consultar deudas, estados de suministro y registrar reportes de fallas o emergencias eléctricas.

## Entidades Principales
- **User**: Clientes registrados con DNI.
- **Supply**: Suministros eléctricos (medidores) vinculados a predios.
- **Ticket**: Registros de atención por fallas o reclamos.
- **EnosaInfo**: Información corporativa y de emergencia.

## Herramientas (Tools)
### Lectura (Read)
- `get_enosa_info`: Datos de contacto y sedes.
- `get_supply_details`: Deuda y estado de conexión.
- `get_ticket`: Seguimiento de reportes.
- `search_supplies_by_dni`: Búsqueda de suministros por cliente.

### Escritura (Write)
- `create_ticket`: Registro de fallas y peligros públicos.

## Política Operativa
- Validación obligatoria de DNI para cualquier gestión.
- Priorización inmediata de tickets tipo `public_hazard`.
- Prohibición de inventar datos no existentes en la base de datos.