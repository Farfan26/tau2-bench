from tau2.environment.toolkit import ToolKitBase, is_tool, ToolType

class DivemotorTools(ToolKitBase):

    @is_tool(ToolType.READ)
    def get_cliente(self, cliente_id: str):
        return self.db.users.get(cliente_id)

    @is_tool(ToolType.READ)
    def buscar_vehiculos(self, tipo: str):
        return [v for v in self.db.vehiculos.values() if tipo.lower() in v.tipo.lower()]

    @is_tool(ToolType.WRITE)
    def crear_cotizacion(self, cliente_id: str, vehiculo_id: str):
        cliente = self.db.users.get(cliente_id)
        vehiculo = self.db.vehiculos.get(vehiculo_id)

        if not cliente or not vehiculo:
            return "Error: cliente o vehiculo no existe"

        if cliente["presupuesto"] < vehiculo["precio"]:
            return "Error: presupuesto insuficiente"

        if vehiculo["stock"] <= 0:
            return "Error: sin stock"

        cot_id = f"cot_{len(self.db.cotizaciones)+1}"

        cot = {
            "id": cot_id,
            "cliente_id": cliente_id,
            "vehiculo_id": vehiculo_id,
            "precio_final": vehiculo["precio"],
            "estado": "pendiente"
        }

        self.db.cotizaciones[cot_id] = cot
        return cot

    @is_tool(ToolType.WRITE)
    def aprobar_cotizacion(self, cotizacion_id: str):
        cot = self.db.cotizaciones.get(cotizacion_id)

        if not cot:
            return "Error: no existe"

        cot["estado"] = "aprobada"
        return cot

    @is_tool(ToolType.WRITE)
    def crear_pedido(self, cotizacion_id: str):
        cot = self.db.cotizaciones.get(cotizacion_id)

        if not cot:
            return "Error: cotizacion no existe"

        if cot["estado"] != "aprobada":
            return "Error: cotizacion no aprobada"

        vehiculo = self.db.vehiculos.get(cot["vehiculo_id"])

        if vehiculo["stock"] <= 0:
            return "Error: sin stock"

        vehiculo["stock"] -= 1

        ped_id = f"ped_{len(self.db.pedidos)+1}"

        ped = {
            "id": ped_id,
            "cotizacion_id": cotizacion_id,
            "estado": "confirmado"
        }

        self.db.pedidos[ped_id] = ped
        return ped

    @is_tool(ToolType.WRITE)
    def cancelar_pedido(self, pedido_id: str):
        ped = self.db.pedidos.get(pedido_id)

        if not ped:
            return "Error: pedido no existe"

        ped["estado"] = "cancelado"
        return ped