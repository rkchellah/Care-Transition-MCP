"""
Tool: generate_transition_brief
--------------------------------
The core tool. Pulls patient FHIR context from Po and uses
Groq (LLaMA 3.3 70B) to generate a structured handoff document.

Fix (May 2026):
    Conditions are now split into clinical vs social before hitting the LLM.
    This prevents the SDoH Trap — where social history (employment, education)
    gets weighted the same as active medical problems in the output.

    Vitals are fetched separately from labs so the LLM can speak to
    the patient's current physiological state, not just historical data.
"""

from typing import Optional
from fhir.client import (
    get_patient_conditions,
    get_patient_encounters,
    get_patient_observations,
)
from llm.groq_client import generate
from tools.patient_summary import get_patient_summary


def _split_conditions(raw_conditions: list) -> tuple[list, list]:
    """
    Split FHIR Condition resources into clinical problems and social history.

    FHIR R4 Condition.category codes:
        social-history       -> psychosocial context section (end of brief)
        problem-list-item    -> active clinical problems
        encounter-diagnosis  -> active clinical problems

    If a condition has no category, it defaults to clinical.

    Returns:
        clinical: list of condition text strings
        social:   list of social history text strings
    """
    clinical = []
    social = []

    for condition in raw_conditions:
        category_codes = [
            coding.get("code", "")
            for cat in condition.get("category", [])
            for coding in cat.get("coding", [])
        ]

        text = (
            condition.get("code", {}).get("text")
            or condition.get("code", {})
            .get("coding", [{}])[0]
            .get("display", "Unknown")
        )

        if "social-history" in category_codes:
            social.append(text)
        else:
            clinical.append(text)

    return clinical, social


async def generate_transition_brief(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
    transition_type: str = "hospital-to-home",
) -> dict:
    summary = await get_patient_summary(fhir_url, patient_id, fhir_token)
    raw_conditions = await get_patient_conditions(fhir_url, patient_id, fhir_token)
    encounters = await get_patient_encounters(fhir_url, patient_id, fhir_token, limit=3)
    vitals = await get_patient_observations(fhir_url, patient_id, fhir_token, category="vital-signs")
    labs = await get_patient_observations(fhir_url, patient_id, fhir_token, category="laboratory")

    clinical_conditions, social_conditions = _split_conditions(raw_conditions)

    # Most recent vitals — clinician needs current physiological state
    vitals_text = ""
    for obs in vitals[:3]:
        name = obs.get("code", {}).get("text", "Vital")
        value = obs.get("valueQuantity", {})
        val_str = f"{value.get('value', '?')} {value.get('unit', '')}".strip()
        date = obs.get("effectiveDateTime", "Unknown date")
        vitals_text += f"- {name}: {val_str} ({date})\n"

    # Most recent 5 labs
    lab_text = ""
    for obs in labs[:5]:
        name = obs.get("code", {}).get("text", "Lab")
        value = obs.get("valueQuantity", {})
        val_str = f"{value.get('value', '?')} {value.get('unit', '')}".strip()
        date = obs.get("effectiveDateTime", "Unknown date")
        interp = (
            obs.get("interpretation", [{}])[0]
            .get("coding", [{}])[0]
            .get("code", "")
        )
        flag = " [ABNORMAL]" if interp in ("H", "L", "HH", "LL", "A") else ""
        lab_text += f"- {name}: {val_str} ({date}){flag}\n"

    # Pull primary admission reason from the most recent encounter
    primary_reason = "Not recorded"
    encounter_history = ""
    for enc in encounters:
        reason = enc.get("reasonCode", [{}])[0].get("text", "")
        enc_type = enc.get("type", [{}])[0].get("text", "Visit")
        enc_date = enc.get("period", {}).get("start", "Unknown date")
        if reason and primary_reason == "Not recorded":
            primary_reason = reason
        encounter_history += f"- {enc_type} on {enc_date}: {reason or 'No reason recorded'}\n"

    # Explicit warning when no medications are on record
    # "None recorded" is ambiguous — could mean no meds or a data gap
    if summary["current_medications"]:
        medications_text = "\n".join(
            f"- {m['medication']}: {m['dosage']}"
            for m in summary["current_medications"]
        )
    else:
        medications_text = (
            "No active medications on record. "
            "Verify with clinical team: this may reflect a data gap, "
            "not the patient's actual medication status."
        )

    prompt = f"""
You are a clinical nurse writing a handover brief for the receiving care team.
Write in the order a clinician actually needs the information.
Use only the data provided. Do not add clinical details not in the data.

PATIENT: {summary['name']} | DOB: {summary['birth_date']}
TRANSITION: {transition_type}

---

Write the handover in this exact order and use these section headings:

1. REASON FOR ADMISSION
State why this patient was admitted. Use this: {primary_reason}

2. CURRENT CLINICAL STATUS
Summarise the patient's physiological stability based on the vitals below.
If vitals are missing, write: "Recent vitals not available — verify before transfer."

RECENT VITALS:
{vitals_text or "No recent vitals recorded"}

3. ACTIVE MEDICAL PROBLEMS
List clinical diagnoses only. Do not include social history here.
{chr(10).join(f"- {c}" for c in clinical_conditions) or "None recorded"}

4. MEDICATIONS AT DISCHARGE
{medications_text}

5. RECENT LABS AND PENDING RESULTS
{lab_text or "No recent labs recorded"}

6. FOLLOW-UP REQUIRED
{encounter_history or "No recent encounters recorded"}

7. PSYCHOSOCIAL CONTEXT AND DISCHARGE BARRIERS
Social history relevant to discharge planning only.
This is background context — not an active medical problem.
{chr(10).join(f"- {s}" for s in social_conditions) or "No social history recorded"}

---

Keep the total brief under 450 words. Be direct. Write like a nurse handing over to the next shift.
"""

    brief_text = await generate(prompt, temperature=0.1)

    return {
        "patient_id": patient_id,
        "patient_name": summary["name"],
        "transition_type": transition_type,
        "brief": brief_text,
    }