import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.healthcare_macalupu.data_model import (
    SIC,
    Doctor,
    InterconsultaDB,
    Patient,
)
from tau2.domains.healthcare_macalupu.environment import get_environment
from tau2.environment.environment import Environment

# =============================================================================
# Fixtures — base de datos mínima para pruebas
# =============================================================================


@pytest.fixture
def healthcare_macalupu_db() -> InterconsultaDB:
    """
    Minimal in-memory DB covering all states and edge cases needed for testing.

    Doctors:
        - 11111111-1 → CESFAM Test A
        - 22222222-2 → CESFAM Test B

    Patients:
        - RUN 11111111-1 → CESFAM Test A, birth 1970-01-01  (has SICs in multiple states)
        - RUN 22222222-2 → CESFAM Test B, birth 1985-06-15  (no SICs)
        - RUN 33333333-3 → CESFAM Test A, birth 1960-03-22  (has a 'citada' SIC)

    SICs:
        SIC-T01 → borrador       (patient 11111111-1, Oftalmología)
        SIC-T02 → enviada        (patient 11111111-1, Traumatología)
        SIC-T03 → pendiente_de_citacion (patient 11111111-1, Cardiología)
        SIC-T04 → atendida       (patient 11111111-1, Salud Mental)    ← non-cancellable
        SIC-T05 → no_pertinente  (patient 11111111-1, Medicina Interna) ← non-cancellable
        SIC-T06 → citada         (patient 33333333-3, Oftalmología)    ← for assert_sic_not_sent
    """
    doctors = {
        "11111111-1": Doctor(
            rut="11111111-1",
            name="Dr. Médico Test A",
            cesfam="CESFAM Test A",
        ),
        "22222222-2": Doctor(
            rut="22222222-2",
            name="Dra. Médica Test B",
            cesfam="CESFAM Test B",
        ),
    }

    patients = {
        "11111111-1": Patient(
            run="11111111-1",
            name="Paciente Con SICs",
            birth_date="1970-01-01",
            cesfam="CESFAM Test A",
            sic_ids=["SIC-T01", "SIC-T02", "SIC-T03", "SIC-T04", "SIC-T05"],
        ),
        "22222222-2": Patient(
            run="22222222-2",
            name="Paciente Sin SICs",
            birth_date="1985-06-15",
            cesfam="CESFAM Test B",
            sic_ids=[],
        ),
        "33333333-3": Patient(
            run="33333333-3",
            name="Paciente Con Cita",
            birth_date="1960-03-22",
            cesfam="CESFAM Test A",
            sic_ids=["SIC-T06"],
        ),
    }

    def _base_sic(**kwargs) -> SIC:
        defaults = dict(
            doctor_rut="11111111-1",
            patient_run="11111111-1",
            cie10_code="H26.9",
            cie10_description="Catarata no especificada",
            reason="Motivo de prueba",
            priority="P2",
            attached_exams=["a-001"],
            is_ges=False,
            created_date="2025-06-01",
            appointment_date=None,
            appointment_location=None,
        )
        defaults.update(kwargs)
        return SIC(**defaults)

    sics = {
        "SIC-T01": _base_sic(
            sic_id="SIC-T01", specialty="Oftalmología", status="borrador"
        ),
        "SIC-T02": _base_sic(
            sic_id="SIC-T02", specialty="Traumatología", status="enviada"
        ),
        "SIC-T03": _base_sic(
            sic_id="SIC-T03", specialty="Cardiología", status="pendiente_de_citacion"
        ),
        "SIC-T04": _base_sic(
            sic_id="SIC-T04", specialty="Salud Mental", status="atendida"
        ),
        "SIC-T05": _base_sic(
            sic_id="SIC-T05", specialty="Medicina Interna", status="no_pertinente"
        ),
        "SIC-T06": _base_sic(
            sic_id="SIC-T06",
            specialty="Oftalmología",
            status="citada",
            patient_run="33333333-3",
            appointment_date="2025-07-10 09:00",
            appointment_location="Hospital de Prueba",
        ),
    }

    return InterconsultaDB(doctors=doctors, patients=patients, sics=sics, analyses={})


@pytest.fixture
def environment(healthcare_macalupu_db: InterconsultaDB) -> Environment:
    return get_environment(healthcare_macalupu_db)


# =============================================================================
# ToolCall fixtures
# =============================================================================


@pytest.fixture
def identificar_medico_call() -> ToolCall:
    return ToolCall(
        id="tc-1", name="identificar_medico", arguments={"rut": "11111111-1"}
    )


@pytest.fixture
def identificar_paciente_call() -> ToolCall:
    return ToolCall(
        id="tc-2",
        name="identificar_paciente",
        arguments={"run": "11111111-1"},
    )


@pytest.fixture
def buscar_sics_paciente_call() -> ToolCall:
    return ToolCall(
        id="tc-4", name="buscar_sics_paciente", arguments={"run": "11111111-1"}
    )


@pytest.fixture
def crear_sic_call() -> ToolCall:
    return ToolCall(
        id="tc-5",
        name="crear_sic",
        arguments={
            "doctor_rut": "11111111-1",
            "patient_run": "11111111-1",
            "specialty": "Otorrinolaringología",
            "cie10_code": "H91.9",
            "cie10_description": "Hipoacusia no especificada",
            "reason": "Paciente con hipoacusia progresiva bilateral.",
            "priority": "P2",
            "attached_exams": ["a-001"],
            "is_ges": False,
        },
    )


@pytest.fixture
def enviar_sic_call() -> ToolCall:
    return ToolCall(id="tc-6", name="enviar_sic", arguments={"sic_id": "SIC-T01"})


@pytest.fixture
def anular_sic_call() -> ToolCall:
    return ToolCall(id="tc-7", name="anular_sic", arguments={"sic_id": "SIC-T01"})


# =============================================================================
# Tests — identificar_medico
# =============================================================================


class TestIdentificarMedico:
    def test_medico_existente_retorna_perfil(
        self, environment: Environment, identificar_medico_call: ToolCall
    ):
        response = environment.get_response(identificar_medico_call)
        assert not response.error
        medico = environment.tools.db.doctors["11111111-1"]
        assert medico.rut == "11111111-1"
        assert medico.cesfam == "CESFAM Test A"

    def test_rut_inexistente_retorna_error(
        self, environment: Environment, identificar_medico_call: ToolCall
    ):
        identificar_medico_call.arguments["rut"] = "99999999-9"
        response = environment.get_response(identificar_medico_call)
        assert response.error


# =============================================================================
# Tests — identificar_paciente
# =============================================================================


class TestIdentificarPaciente:
    def test_paciente_existente_retorna_perfil(
        self, environment: Environment, identificar_paciente_call: ToolCall
    ):
        response = environment.get_response(identificar_paciente_call)
        assert not response.error
        paciente = environment.tools.db.patients["11111111-1"]
        assert paciente.run == "11111111-1"
        assert paciente.cesfam == "CESFAM Test A"

    def test_run_inexistente_retorna_error(
        self, environment: Environment, identificar_paciente_call: ToolCall
    ):
        identificar_paciente_call.arguments["run"] = "99999999-9"
        response = environment.get_response(identificar_paciente_call)
        assert response.error

    def test_fecha_nacimiento_incorrecta_retorna_error(
        self, environment: Environment, identificar_paciente_call: ToolCall
    ):
        identificar_paciente_call.arguments["birth_date"] = "1999-12-31"
        response = environment.get_response(identificar_paciente_call)
        assert response.error


# =============================================================================
# Tests — buscar_sics_paciente
# =============================================================================


class TestBuscarSicsPaciente:
    def test_paciente_con_sics_retorna_lista(
        self, environment: Environment, buscar_sics_paciente_call: ToolCall
    ):
        response = environment.get_response(buscar_sics_paciente_call)
        assert not response.error
        # Patient 11111111-1 has 5 SICs
        sics = environment.tools.buscar_sics_paciente("11111111-1")
        assert len(sics) == 5
        sic_ids = {s.sic_id for s in sics}
        assert "SIC-T01" in sic_ids
        assert "SIC-T04" in sic_ids

    def test_paciente_sin_sics_retorna_lista_vacia(
        self, environment: Environment, buscar_sics_paciente_call: ToolCall
    ):
        buscar_sics_paciente_call.arguments["run"] = "22222222-2"
        response = environment.get_response(buscar_sics_paciente_call)
        assert not response.error
        sics = environment.tools.buscar_sics_paciente("22222222-2")
        assert sics == []

    def test_run_inexistente_retorna_error(
        self, environment: Environment, buscar_sics_paciente_call: ToolCall
    ):
        buscar_sics_paciente_call.arguments["run"] = "99999999-9"
        response = environment.get_response(buscar_sics_paciente_call)
        assert response.error


# =============================================================================
# Tests — crear_sic
# =============================================================================


class TestCrearSic:
    def test_crear_sic_valida(self, environment: Environment, crear_sic_call: ToolCall):
        sic_count_before = len(environment.tools.db.sics)
        patient_sic_count_before = len(
            environment.tools.db.patients["11111111-1"].sic_ids
        )

        response = environment.get_response(crear_sic_call)
        assert not response.error

        # One new SIC must have been added to the DB
        assert len(environment.tools.db.sics) == sic_count_before + 1

        # The new SIC must be linked to the patient
        assert (
            len(environment.tools.db.patients["11111111-1"].sic_ids)
            == patient_sic_count_before + 1
        )

        # Retrieve new SIC by its generated ID
        new_sic_id = f"SIC-{sic_count_before + 1:03d}"
        new_sic = environment.tools.db.sics[new_sic_id]

        assert new_sic.status == "borrador"
        assert new_sic.specialty == "Otorrinolaringología"
        assert new_sic.cie10_code == "H91.9"
        assert new_sic.priority == "P2"
        assert new_sic.is_ges is False
        assert new_sic.attached_exams == ["a-001"]
        assert new_sic.appointment_date is None
        assert new_sic.appointment_location is None

    def test_doctor_inexistente_retorna_error(
        self, environment: Environment, crear_sic_call: ToolCall
    ):
        crear_sic_call.arguments["doctor_rut"] = "99999999-9"
        response = environment.get_response(crear_sic_call)
        assert response.error

    def test_paciente_inexistente_retorna_error(
        self, environment: Environment, crear_sic_call: ToolCall
    ):
        crear_sic_call.arguments["patient_run"] = "99999999-9"
        response = environment.get_response(crear_sic_call)
        assert response.error

    def test_nueva_sic_no_altera_otras_sics(
        self, environment: Environment, crear_sic_call: ToolCall
    ):
        """Creating a SIC must not change the state of pre-existing SICs."""
        original_status = environment.tools.db.sics["SIC-T01"].status
        environment.get_response(crear_sic_call)
        assert environment.tools.db.sics["SIC-T01"].status == original_status


# =============================================================================
# Tests — enviar_sic
# =============================================================================


class TestEnviarSic:
    def test_enviar_sic_en_borrador(
        self, environment: Environment, enviar_sic_call: ToolCall
    ):
        assert environment.tools.db.sics["SIC-T01"].status == "borrador"
        response = environment.get_response(enviar_sic_call)
        assert not response.error
        assert environment.tools.db.sics["SIC-T01"].status == "enviada"

    def test_sic_inexistente_retorna_error(
        self, environment: Environment, enviar_sic_call: ToolCall
    ):
        enviar_sic_call.arguments["sic_id"] = "SIC-INEXISTENTE"
        response = environment.get_response(enviar_sic_call)
        assert response.error

    def test_sic_ya_enviada_retorna_error(
        self, environment: Environment, enviar_sic_call: ToolCall
    ):
        """Sending a SIC that is already 'enviada' must fail."""
        enviar_sic_call.arguments["sic_id"] = "SIC-T02"  # already 'enviada'
        response = environment.get_response(enviar_sic_call)
        assert response.error

    def test_sic_atendida_retorna_error(
        self, environment: Environment, enviar_sic_call: ToolCall
    ):
        """Sending a SIC that is 'atendida' must fail."""
        enviar_sic_call.arguments["sic_id"] = "SIC-T04"
        response = environment.get_response(enviar_sic_call)
        assert response.error

    def test_sic_pendiente_citacion_retorna_error(
        self, environment: Environment, enviar_sic_call: ToolCall
    ):
        enviar_sic_call.arguments["sic_id"] = "SIC-T03"  # 'pendiente_de_citacion'
        response = environment.get_response(enviar_sic_call)
        assert response.error


# =============================================================================
# Tests — anular_sic
# =============================================================================


class TestAnularSic:
    @pytest.mark.parametrize("sic_id", ["SIC-T01", "SIC-T02", "SIC-T03"])
    def test_anular_estados_permitidos(
        self, environment: Environment, anular_sic_call: ToolCall, sic_id: str
    ):
        """SICs in borrador, enviada, or pendiente_de_citacion must be cancellable."""
        anular_sic_call.arguments["sic_id"] = sic_id
        response = environment.get_response(anular_sic_call)
        assert not response.error
        assert environment.tools.db.sics[sic_id].status == "anulada"

    @pytest.mark.parametrize("sic_id", ["SIC-T04", "SIC-T05", "SIC-T06"])
    def test_anular_estados_no_permitidos_retorna_error(
        self,
        environment: Environment,
        anular_sic_call: ToolCall,
        sic_id: str,
        healthcare_macalupu_db: InterconsultaDB,
    ):
        """SICs in atendida, no_pertinente, or citada must NOT be cancellable."""
        # SIC-T06 belongs to patient 33333333-3, add it to db sics lookup manually
        anular_sic_call.arguments["sic_id"] = sic_id
        response = environment.get_response(anular_sic_call)
        assert response.error
        # The status must remain unchanged
        assert environment.tools.db.sics[sic_id].status != "anulada"

    def test_sic_inexistente_retorna_error(
        self, environment: Environment, anular_sic_call: ToolCall
    ):
        anular_sic_call.arguments["sic_id"] = "SIC-INEXISTENTE"
        response = environment.get_response(anular_sic_call)
        assert response.error


# =============================================================================
# Tests — assert_sic_status
# =============================================================================


class TestAssertSicStatus:
    def test_estado_correcto_retorna_true(self, environment: Environment):
        assert environment.tools.assert_sic_status("SIC-T01", "borrador") is True
        assert environment.tools.assert_sic_status("SIC-T02", "enviada") is True
        assert (
            environment.tools.assert_sic_status("SIC-T03", "pendiente_de_citacion")
            is True
        )
        assert environment.tools.assert_sic_status("SIC-T04", "atendida") is True
        assert environment.tools.assert_sic_status("SIC-T05", "no_pertinente") is True
        assert environment.tools.assert_sic_status("SIC-T06", "citada") is True

    def test_estado_incorrecto_retorna_false(self, environment: Environment):
        assert environment.tools.assert_sic_status("SIC-T01", "enviada") is False
        assert environment.tools.assert_sic_status("SIC-T02", "borrador") is False
        assert environment.tools.assert_sic_status("SIC-T04", "anulada") is False

    def test_sic_inexistente_lanza_excepcion(self, environment: Environment):
        with pytest.raises(ValueError):
            environment.tools.assert_sic_status("SIC-INEXISTENTE", "borrador")

    def test_refleja_cambio_de_estado_tras_envio(self, environment: Environment):
        """assert_sic_status must reflect DB mutations made by other tools."""
        assert environment.tools.assert_sic_status("SIC-T01", "borrador") is True
        environment.tools.enviar_sic("SIC-T01")
        assert environment.tools.assert_sic_status("SIC-T01", "borrador") is False
        assert environment.tools.assert_sic_status("SIC-T01", "enviada") is True


# =============================================================================
# Tests — assert_sic_not_sent
# =============================================================================


class TestAssertSicNotSent:
    def test_paciente_sin_sics_enviadas_retorna_true(self, environment: Environment):
        # Patient 22222222-2 has no SICs at all
        assert environment.tools.assert_sic_not_sent("22222222-2") is True

    def test_sic_en_estado_citada_retorna_false(self, environment: Environment):
        # Patient 33333333-3 has SIC-T06 in 'citada' state
        assert environment.tools.assert_sic_not_sent("33333333-3") is False

    def test_filtro_por_especialidad_coincidente_retorna_false(
        self, environment: Environment
    ):
        # Patient 33333333-3 has a 'citada' Oftalmología SIC
        assert (
            environment.tools.assert_sic_not_sent("33333333-3", "Oftalmología") is False
        )

    def test_filtro_por_especialidad_no_coincidente_retorna_true(
        self, environment: Environment
    ):
        # Patient 33333333-3 has no sent Traumatología SIC
        assert (
            environment.tools.assert_sic_not_sent("33333333-3", "Traumatología") is True
        )

    def test_paciente_solo_con_sics_no_enviadas_retorna_true(
        self, environment: Environment
    ):
        # Patient 11111111-1 has SIC-T01 (borrador) — not a 'sent' state
        assert (
            environment.tools.assert_sic_not_sent("11111111-1", "Oftalmología") is True
        )

    def test_run_inexistente_retorna_true(self, environment: Environment):
        # Non-existent patient → nothing was sent → True
        assert environment.tools.assert_sic_not_sent("99999999-9") is True

    def test_refleja_envio_realizado_por_enviar_sic(self, environment: Environment):
        """assert_sic_not_sent must return False after enviar_sic is called."""
        # Before sending: SIC-T01 is borrador → not sent
        assert (
            environment.tools.assert_sic_not_sent("11111111-1", "Oftalmología") is True
        )
        environment.tools.enviar_sic("SIC-T01")
        # After sending: SIC-T01 is enviada → now considered sent
        assert (
            environment.tools.assert_sic_not_sent("11111111-1", "Oftalmología") is False
        )
