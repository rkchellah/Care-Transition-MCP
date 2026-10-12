"""
Tool: medication_reconciliation
---------------------------------
Compares current active medications against encounter history
and flags discrepancies. Top cause of post-discharge harm.
"""

import re
from typing import Optional
from fhir.client import DataUnavailable, get_patient_medications, get_patient_encounters
from llm.groq_client import generate
from tools.patient_summary import (
    ACCESS_DENIED_RULE,
    describe,
    get_patient_summary,
    unavailable_notes,
)


async def medication_reconciliation(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
) -> dict:
    medications = await get_patient_medications(fhir_url, patient_id, fhir_token)
    encounters = await get_patient_encounters(fhir_url, patient_id, fhir_token, limit=2)
    summary = await get_patient_summary(fhir_url, patient_id, fhir_token)

    notes = unavailable_notes(summary, medications, encounters)

    current_meds = []
    for m in ([] if isinstance(medications, DataUnavailable) else medications):
        med = m.get("medicationCodeableConcept", {})
        name = med.get("text") or (med.get("coding", [{}])[0].get("display", "Unknown"))
        dosage = m.get("dosageInstruction", [{}])[0].get("text", "Not specified")
        status = m.get("status", "unknown")
        current_meds.append(f"- {name} | Dosage: {dosage} | Status: {status}")

    if isinstance(medications, DataUnavailable):
        meds_text = str(medications)
    else:
        meds_text = "\n".join(current_meds) or "No active medications on record"

    encounter_context = str(encounters) if isinstance(encounters, DataUnavailable) else ""
    for enc in ([] if isinstance(encounters, DataUnavailable) else encounters):
        note = enc.get("text", {}).get("div", "")
        if note:
            clean_note = re.sub(r"<[^>]+>", " ", note).strip()
            encounter_context += f"{clean_note[:300]}\n"

    prompt = f"""
You are a pharmacist reviewing medication reconciliation at discharge.{ACCESS_DENIED_RULE.rstrip() if notes else ""}

PATIENT: {summary['name']}
ACTIVE CONDITIONS: {describe(summary, 'active_conditions', ', '.join, 'None')}

CURRENT MEDICATIONS ON RECORD:
{meds_text}

RECENT ENCOUNTER CONTEXT:
{encounter_context or "No encounter notes available"}

Review and identify:
- DUPLICATE THERAPIES
- MISSING MEDICATIONS (condition exists but no corresponding medication)
- DOSAGE CONCERNS
- DRUG INTERACTIONS

If no issues in a category, write "None identified."
Be specific and clinical. Do not fabricate interactions not supported by the data.
"""

    reconciliation_text = await generate(prompt, temperature=0.1)

    return {
        "patient_id": patient_id,
        "patient_name": summary["name"],
        "medication_count": None if isinstance(medications, DataUnavailable) else len(medications),
        "reconciliation_findings": reconciliation_text,
        "data_unavailable": notes,
    }
