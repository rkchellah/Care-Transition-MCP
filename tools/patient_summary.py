"""
Tool: get_patient_summary
-------------------------
Pulls core patient demographics, active conditions,
and current medications from the Po FHIR server.
Uses SHARP context (fhir_url, fhir_token) from Po.
"""

from typing import Callable, Optional
from fhir.client import (
    DataUnavailable,
    get_patient,
    get_patient_conditions,
    get_patient_medications,
)

# Added to a prompt only when some data could not be read
ACCESS_DENIED_RULE = """
IMPORTANT: Any line marked "ACCESS DENIED" means that data exists but could not be read.
Repeat that line exactly in the relevant section. Never describe it as none, empty,
not recorded, or not present.
"""


def describe(summary: dict, field: str, render: Callable[[list], str], empty: str) -> str:
    """
    Text for a summary list field: the rendered items, the `empty` text if the
    list is genuinely empty, or the ACCESS DENIED line if it could not be read.
    """
    value = summary[field]
    if value is None:
        return summary["data_unavailable"][field]
    return render(value) or empty


def unavailable_notes(summary: dict, *results) -> list[str]:
    """Collect ACCESS DENIED lines from a summary and raw FHIR results, without duplicates."""
    notes = list(summary["data_unavailable"].values())
    notes += [str(r) for r in results if isinstance(r, DataUnavailable)]
    return list(dict.fromkeys(notes))


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
        and data_unavailable. A list that could not be read (401/403) is None,
        with its ACCESS DENIED line in data_unavailable under the same key.
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

    data_unavailable = {}

    # Extract active conditions
    if isinstance(conditions, DataUnavailable):
        condition_list = None
        data_unavailable["active_conditions"] = str(conditions)
    else:
        condition_list = []
        for c in conditions:
            code = c.get("code", {})
            text = code.get("text") or (code.get("coding", [{}])[0].get("display", "Unknown condition"))
            condition_list.append(text)

    # Extract medications
    if isinstance(medications, DataUnavailable):
        medication_list = None
        data_unavailable["current_medications"] = str(medications)
    else:
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
        "data_unavailable": data_unavailable,
    }
