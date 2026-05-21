from tau2.domains.divemotor_santiago.environment import get_environment


def test_divemotor_tools_sale_flow():
    env = get_environment()

    cliente = env.use_tool("get_cliente", cliente_id="c1")
    assert cliente.id == "c1"

    vehiculos = env.use_tool("buscar_vehiculos", tipo="camion")
    assert len(vehiculos) == 1
    assert vehiculos[0].id == "v1"

    cotizacion = env.use_tool("crear_cotizacion", cliente_id="c1", vehiculo_id="v1")
    assert cotizacion.id == "cot_1"
    assert cotizacion.estado == "pendiente"

    cotizacion = env.use_tool("aprobar_cotizacion", cotizacion_id="cot_1")
    assert cotizacion.estado == "aprobada"

    pedido = env.use_tool("crear_pedido", cotizacion_id="cot_1")
    assert pedido.id == "ped_1"
    assert pedido.estado == "confirmado"
    assert env.tools.db.vehiculos["v1"].stock == 2

    pedido = env.use_tool("cancelar_pedido", pedido_id="ped_1")
    assert pedido.estado == "cancelado"


def test_divemotor_tools_reject_invalid_sale():
    env = get_environment()

    assert (
        env.use_tool("crear_cotizacion", cliente_id="c2", vehiculo_id="v1")
        == "Error: presupuesto insuficiente"
    )
    assert (
        env.use_tool("crear_cotizacion", cliente_id="c1", vehiculo_id="v3")
        == "Error: sin stock"
    )
    assert env.use_tool("crear_pedido", cotizacion_id="cot_1") == (
        "Error: cotizacion no existe"
    )
