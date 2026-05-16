from datetime import date
from typing import Optional

from tau2.domains.healthcare_macalupu.data_model import (
    SIC,
    Analysis,
    Doctor,
    InterconsultaDB,
    Patient,
    Priority,
    SICStatus,
    Specialty,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# States from which a SIC can be cancelled
_CANCELLABLE_STATES: set[SICStatus] = {"borrador", "enviada", "pendiente_de_citacion"}


class InterconsultaTools(ToolKitBase):
    """Tools for the Chilean health referral (interconsulta) domain."""

    db: InterconsultaDB

    def __init__(self, db: InterconsultaDB) -> None:
        super().__init__(db)

    # -------------------------------------------------------------------------
    # READ tools
    # -------------------------------------------------------------------------

    @is_tool(ToolType.READ)
    def identificar_medico(self, rut: str) -> Doctor:
        """
        Retrieve a doctor's profile by their RUT.
        Use this at the start of the conversation to authenticate a doctor.

        Args:
            rut: The doctor's RUT (e.g. '12345678-9').

        Returns:
            The doctor's profile.

        Raises:
            ValueError: If no doctor with that RUT is found.
        """
        if rut not in self.db.doctors:
            raise ValueError(f"No se encontró ningún médico con RUT {rut}.")
        return self.db.doctors[rut]

    @is_tool(ToolType.READ)
    def identificar_paciente(self, run: str) -> Patient:
        """
        Retrieve a patient's profile by their RUN.
        Use this at the start of the conversation to authenticate a patient.

        Args:
            run: The patient's RUN (e.g. '15432876-3').

        Returns:
            The patient's profile.

        Raises:
            ValueError: If no patient is found with that RUN and birth date combination.
        """
        if run not in self.db.patients:
            raise ValueError(f"No se encontró ningún paciente con RUN {run}.")
        return self.db.patients[run]

    @is_tool(ToolType.READ)
    def get_sic(self, sic_id: str) -> SIC:
        """
        Retrieve a specific referral request (SIC) by its ID.

        Args:
            sic_id: The SIC identifier (e.g. 'SIC-001').

        Returns:
            The full SIC record.

        Raises:
            ValueError: If no SIC with that ID is found.
        """
        try:
            return self.db.sics[sic_id]
        except KeyError:
            raise ValueError(f"No se encontró ninguna interconsulta con ID {sic_id}.")

    @is_tool(ToolType.READ)
    def buscar_sics_paciente(self, run: str) -> list[SIC]:
        """
        Retrieve all referral requests (SICs) associated with a patient.
        The agent is responsible for verifying that the caller has permission
        to access this patient's data before invoking this tool.

        Args:
            run: The patient's RUN.

        Returns:
            A list of SIC records for the patient (may be empty).

        Raises:
            ValueError: If no patient with that RUN is found.
        """
        if run not in self.db.patients:
            raise ValueError(f"No se encontró ningún paciente con RUN {run}.")
        patient = self.db.patients[run]
        return [
            self.db.sics[sic_id] for sic_id in patient.sic_ids if sic_id in self.db.sics
        ]

    @is_tool(ToolType.READ)
    def buscar_analisis(self, id: str) -> Analysis:
        """
        Retrieve an analysis by its ID.

        Args:
            id: The analysis ID.

        Returns:
            The analysis record.

        Raises:
            ValueError: If no analysis with that ID is found.
        """
        if id not in self.db.analyses:
            raise ValueError(f"No se encontró ningún análisis con ID {id}.")
        return self.db.analyses[id]

    # -------------------------------------------------------------------------
    # WRITE tools
    # -------------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def crear_sic(
        self,
        doctor_rut: str,
        patient_run: str,
        specialty: Specialty,
        cie10_code: str,
        cie10_description: str,
        reason: str,
        priority: Priority,
        attached_exams: list[str],
        is_ges: bool,
    ) -> SIC:
        """
        Create a new referral request (SIC) in 'borrador' (draft) state.
        The agent must verify all clinical and administrative criteria
        (required exams, GES flag, CESFAM match) BEFORE calling this tool.
        This tool does NOT validate clinical criteria.

        Args:
            doctor_rut: RUT of the requesting doctor.
            patient_run: RUN of the patient being referred.
            specialty: Target medical specialty.
            cie10_code: CIE-10 diagnosis code (e.g. 'H26.9').
            cie10_description: Human-readable CIE-10 description.
            reason: Clinical justification for the referral.
            priority: 'P1' for urgent, 'P2' for non-urgent.
            attached_exams: List of exam's identifiers confirmed as attached.
            is_ges: True if the condition has a GES/AUGE guarantee.

        Returns:
            The newly created SIC in 'borrador' state.

        Raises:
            ValueError: If the doctor or patient is not found.
        """
        if doctor_rut not in self.db.doctors:
            raise ValueError(f"No se encontró ningún médico con RUT {doctor_rut}.")
        if patient_run not in self.db.patients:
            raise ValueError(f"No se encontró ningún paciente con RUN {patient_run}.")

        sic_id = f"SIC-{len(self.db.sics) + 1:03d}"
        sic = SIC(
            sic_id=sic_id,
            patient_run=patient_run,
            doctor_rut=doctor_rut,
            specialty=specialty,
            cie10_code=cie10_code,
            cie10_description=cie10_description,
            reason=reason,
            priority=priority,
            attached_exams=attached_exams,
            status="borrador",
            is_ges=is_ges,
            created_date=date.today().isoformat(),
            appointment_date=None,
            appointment_location=None,
        )

        self.db.sics[sic_id] = sic
        self.db.patients[patient_run].sic_ids.append(sic_id)

        return sic

    @is_tool(ToolType.WRITE)
    def enviar_sic(self, sic_id: str) -> SIC:
        """
        Send a referral request (SIC), transitioning it from 'borrador' to 'enviada'.
        The agent must have already confirmed that all required exams are attached
        and that the patient and doctor belong to the same CESFAM before calling this tool.
        This tool does NOT validate clinical criteria.

        Args:
            sic_id: The ID of the SIC to send.

        Returns:
            The updated SIC in 'enviada' state.

        Raises:
            ValueError: If the SIC is not found or is not in 'borrador' state.
        """
        if sic_id not in self.db.sics:
            raise ValueError(f"No se encontró ninguna interconsulta con ID {sic_id}.")
        sic = self.db.sics[sic_id]
        if sic.status != "borrador":
            raise ValueError(
                f"Solo se pueden enviar interconsultas en estado 'borrador'. "
                f"El estado actual de {sic_id} es '{sic.status}'."
            )
        sic.status = "enviada"
        return sic

    @is_tool(ToolType.WRITE)
    def anular_sic(self, sic_id: str) -> SIC:
        """
        Cancel a referral request (SIC), transitioning it to 'anulada'.
        Only SICs in 'borrador', 'enviada', or 'pendiente_de_citacion' state can be cancelled.

        Args:
            sic_id: The ID of the SIC to cancel.

        Returns:
            The updated SIC in 'anulada' state.

        Raises:
            ValueError: If the SIC is not found or its current state does not allow cancellation.
        """
        if sic_id not in self.db.sics:
            raise ValueError(f"No se encontró ninguna interconsulta con ID {sic_id}.")
        sic = self.db.sics[sic_id]
        if sic.status not in _CANCELLABLE_STATES:
            raise ValueError(
                f"La interconsulta {sic_id} no puede anularse porque está en estado '{sic.status}'. "
                f"Solo se pueden anular interconsultas en estado: {', '.join(_CANCELLABLE_STATES)}."
            )
        sic.status = "anulada"
        return sic

    # -------------------------------------------------------------------------
    # Assert functions (used by the evaluator, not callable by the agent)
    # -------------------------------------------------------------------------

    def assert_sic_status(self, sic_id: str, expected_status: SICStatus) -> bool:
        try:
            return self.db.sics[sic_id].status == expected_status
        except KeyError:
            raise ValueError(f"No se encontró ninguna interconsulta con ID {sic_id}.")

    def assert_sic_not_sent(
        self,
        patient_run: str,
        specialty: Optional[Specialty] = None,
    ) -> bool:
        """
        Check that no SIC for a given patient (and optionally specialty) is in
        'enviada', 'pendiente_de_citacion', or 'citada' state.
        Used to verify the agent did not send a referral it should have blocked.

        Args:
            patient_run: The patient's RUN.
            specialty: Optional specialty filter. If provided, only SICs for
                       that specialty are checked.

        Returns:
            True if no matching SIC has been sent (i.e. the block was effective).
        """
        if patient_run not in self.db.patients:
            return True  # patient doesn't exist → nothing was sent
        sent_statuses = {"enviada", "pendiente_de_citacion", "citada"}
        for sic_id in self.db.patients[patient_run].sic_ids:
            if sic_id not in self.db.sics:
                continue
            sic = self.db.sics[sic_id]
            if specialty is not None and sic.specialty != specialty:
                continue
            if sic.status in sent_statuses:
                return False
        return True
