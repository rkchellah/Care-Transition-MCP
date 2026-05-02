"""
Tool: get_patient_summary
-------------------------
Pulls core patient demographics, active conditions,
and current medications from the Po FHIR server.
Uses SHARP context (fhir_url, fhir_token) from Po.
"""

from typing import Optional
from fhir.client import get_patient, get_patient_conditions, get_patient_medications


async def get_patient_summary(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
) -> dict:
    """
    Fetch and structure a patient's core clinical profile.

    Args:
        fhir_url:   Po workspace FHIR base URL
        patient_id: The FHIR patient resource ID
        fhir_token: Bearer token for FHIR access

    Returns:
        A dictionary with name, birth_date, active_conditions, current_medications
    """
    patient = await get_patient(fhir_url, patient_id, fhir_token)
    conditions = await get_patient_conditions(fhir_url, patient_id, fhir_token)
    medications = await get_patient_medications(fhir_url, patient_id, fhir_token)

    # Extract name
    name_data = patient.get("name", [{}])[0]
    given = " ".join(name_data.get("given", []))
    family = name_data.get("family", "")
    full_name = f"{given} {family}".strip() or "Unknown"

    # Extract birth date
    birth_date = patient.get("birthDate", "Unknown")

    # Extract active conditions
    condition_list = []
    for c in conditions:
        code = c.get("code", {})
        text = code.get("text") or (code.get("coding", [{}])[0].get("display", "Unknown condition"))
        condition_list.append(text)

    # Extract medications
    medication_list = []
    for m in medications:
        med = m.get("medicationCodeableConcept", {})
        text = med.get("text") or (med.get("coding", [{}])[0].get("display", "Unknown medication"))
        dosage = m.get("dosageInstruction", [{}])[0].get("text", "Dosage not specified")
        medication_list.append({"medication": text, "dosage": dosage})

    return {
        "patient_id": patient_id,
        "name": full_name,
        "birth_date": birth_date,
        "active_conditions": condition_list,
        "current_medications": medication_list,
    }
