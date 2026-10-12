"""
Tool: flag_followup_gaps
-------------------------
Checks Po FHIR data for missing follow-up appointments,
abnormal labs, and open medication follow-ups.
"""

from typing import Optional
from fhir.client import DataUnavailable, get_patient_appointments, get_patient_observations
from llm.groq_client import generate
from tools.patient_summary import (
    ACCESS_DENIED_RULE,
    describe,
    get_patient_summary,
    unavailable_notes,
)


async def flag_followup_gaps(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
) -> dict:
    appointments = await get_patient_appointments(fhir_url, patient_id, fhir_token)
    observations = await get_patient_observations(fhir_url, patient_id, fhir_token)
    summary = await get_patient_summary(fhir_url, patient_id, fhir_token)

    notes = unavailable_notes(summary, appointments, observations)

    appointment_list = []
    for appt in ([] if isinstance(appointments, DataUnavailable) else appointments):
        appt_type = appt.get("serviceType", [{}])[0].get("text", "Appointment")
        appt_date = appt.get("start", "Date not set")
        status = appt.get("status", "unknown")
        appointment_list.append(f"- {appt_type} | {appt_date} | Status: {status}")

    abnormal_labs = []
    for obs in ([] if isinstance(observations, DataUnavailable) else observations):
        interpretation = obs.get("interpretation", [{}])[0].get("coding", [{}])[0].get("code", "")
        if interpretation in ("H", "L", "HH", "LL", "A"):
            name = obs.get("code", {}).get("text", "Lab")
            value = obs.get("valueQuantity", {})
            val_str = f"{value.get('value', '?')} {value.get('unit', '')}".strip()
            abnormal_labs.append(f"- {name}: {val_str} [ABNORMAL: {interpretation}]")

    if isinstance(appointments, DataUnavailable):
        appointments_text = str(appointments)
    else:
        appointments_text = "\n".join(appointment_list) or "No upcoming appointments found"

    if isinstance(observations, DataUnavailable):
        labs_text = str(observations)
    else:
        labs_text = "\n".join(abnormal_labs) or "No abnormal labs detected"

    conditions_text = describe(summary, "active_conditions", ", ".join, "None")
    medications_text = describe(
        summary,
        "current_medications",
        lambda meds: ", ".join(m["medication"] for m in meds),
        "None",
    )

    prompt = f"""
You are a care coordinator reviewing a patient's discharge readiness.
Identify gaps in follow-up care. Flag anything that could cause a readmission if missed.{ACCESS_DENIED_RULE.rstrip() if notes else ""}

PATIENT: {summary['name']}
ACTIVE CONDITIONS: {conditions_text}
CURRENT MEDICATIONS: {medications_text}

BOOKED APPOINTMENTS:
{appointments_text}

ABNORMAL LAB VALUES:
{labs_text}

List gaps under:
- MISSING APPOINTMENTS
- UNRESOLVED LAB RESULTS
- MEDICATION FOLLOW-UP NEEDED
- OTHER GAPS

If no gaps in a category, write "None identified." Be direct.
"""

    gaps_text = await generate(prompt, temperature=0.1)

    return {
        "patient_id": patient_id,
        "patient_name": summary["name"],
        "gaps": gaps_text,
        "abnormal_lab_count": None if isinstance(observations, DataUnavailable) else len(abnormal_labs),
        "upcoming_appointments": None if isinstance(appointments, DataUnavailable) else len(appointment_list),
        "data_unavailable": notes,
    }
