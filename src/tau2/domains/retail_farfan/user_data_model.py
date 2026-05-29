from pydantic import BaseModel, Field  # type: ignore
from typing import List


class SMSMessage(BaseModel):
    user_id: str
    code: str


class RetailUserDB(BaseModel):
    """Base de datos para el simulador de usuario."""

    # Se usa Field(default_factory=list) para evitar errores de referencia
    sms_messages: List[SMSMessage] = Field(default_factory=list)
