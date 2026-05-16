from pydoc import describe
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.healthcare_macalupu.utils import HEALTHCARE_DB_PATH
from tau2.environment.db import DB

SICStatus = Literal[
    "borrador",
    "enviada",
    "pendiente_de_citacion",
    "citada",
    "devuelta",
    "no_pertinente",
    "atendida",
    "anulada",
]

Priority = Literal["P1", "P2"]

Specialty = Literal[
    "Oftalmología",
    "Otorrinolaringología",
    "Traumatología",
    "Odontología Especializada",
    "Medicina Interna",
    "Cardiología",
    "Salud Mental",
]


class SIC(BaseModel):
    """Solicitud de Interconsulta."""

    sic_id: str = Field(description="Unique identifier for the referral request")
    patient_run: str = Field(description="Patient's RUN (national ID)")
    doctor_rut: str = Field(description="Requesting doctor's RUT")
    specialty: Specialty = Field(description="Target medical specialty")
    cie10_code: str = Field(description="CIE-10 diagnosis code")
    cie10_description: str = Field(description="CIE-10 diagnosis description")
    reason: str = Field(description="Clinical reason for referral")
    priority: Priority = Field(
        description="Priority level: P1 (urgent) or P2 (non-urgent)"
    )
    attached_exams: List[str] = Field(description="List of attached exam's identifiers")
    status: SICStatus = Field(description="Current status of the referral request")
    is_ges: bool = Field(
        description="Whether the condition is covered by GES guarantee"
    )
    created_date: str = Field(description="Creation date in YYYY-MM-DD format")
    appointment_date: Optional[str] = Field(
        None, description="Appointment date in YYYY-MM-DD HH:MM format, if assigned"
    )
    appointment_location: Optional[str] = Field(
        None, description="Name of the establishment where the appointment is scheduled"
    )


class Doctor(BaseModel):
    """APS doctor (primary care physician)."""

    rut: str = Field(description="Doctor's RUT (unique identifier)")
    name: str = Field(description="Doctor's full name")
    cesfam: str = Field(description="CESFAM the doctor belongs to")


class Patient(BaseModel):
    """Patient registered in the primary care network."""

    run: str = Field(description="Patient's RUN (unique national identifier)")
    name: str = Field(description="Patient's full name")
    birth_date: str = Field(description="Date of birth in YYYY-MM-DD format")
    cesfam: str = Field(description="CESFAM where the patient is enrolled")
    sic_ids: List[str] = Field(
        description="List of referral request IDs for this patient"
    )


class Analysis(BaseModel):
    """Analysis of a patient's referral request."""

    id: str = Field(description="Analysis ID")
    description: str = Field(description="Analysis description")
    patient_run: str = Field(description="Patient's RUN (unique national identifier)")
    details: Optional[str] = Field(description="Analysis details")


class InterconsultaDB(DB):
    """Database for the Chilean health referral agent."""

    doctors: Dict[str, Doctor] = Field(
        description="Dictionary of doctors indexed by RUT"
    )
    patients: Dict[str, Patient] = Field(
        description="Dictionary of patients indexed by RUN"
    )
    sics: Dict[str, SIC] = Field(
        description="Dictionary of referral requests indexed by SIC ID"
    )
    analyses: Dict[str, Analysis] = Field(
        description="Dictionary of analyses indexed by analysis ID"
    )


def get_db():
    return InterconsultaDB.load(HEALTHCARE_DB_PATH)
