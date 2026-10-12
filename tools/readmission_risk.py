"""
Tool: readmission_risk_narrative
----------------------------------
Reads the full FHIR picture from Po and generates a
plain-language risk narrative. This is where the AI factor
is clearest — no rule-based system synthesizes this.

Fix (May 2026):
    Clinical conditions and social history are now passed to the LLM
    as separate, explicitly labelled data blocks.

    Before this fix, the flat conditions list caused the LLM to treat
    "full-time employment" with the same weight as "Myocardial Infarction"
    when determining risk level — because they looked identical in the prompt.

    Now:
        Clinical conditions  -> drive the RISK LEVEL and KEY CLINICAL RISK DRIVERS
        Social conditions    -> populate CONTEXTUAL BARRIERS only
"""

from typing import Optional
from fhir.client import (
    DataUnavailable,
    get_patient_conditions,
    get_patient_encounters,
    get_patient_observations,
)
from llm.groq_client import generate
from tools.patient_summary import ACCESS_DENIED_RULE, get_patient_summary, unavailable_notes
from tools.transition_brief import _split_conditions


async def readmission_risk_narrative(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
) -> dict:
    summary = await get_patient_summary(fhir_url, patient_id, fhir_token)
    raw_conditions = await get_patient_conditions(fhir_url, patient_id, fhir_token)
    encounters = await get_patient_encounters(fhir_url, patient_id, fhir_token, limit=5)
    observations = await get_patient_observations(fhir_url, patient_id, fhir_token)

    notes = unavailable_notes(summary, raw_conditions, encounters, observations)

    if isinstance(raw_conditions, DataUnavailable):
        clinical_text = social_text = str(raw_conditions)
    else:
        clinical_conditions, social_conditions = _split_conditions(raw_conditions)
        clinical_text = "\n".join(f"- {c}" for c in clinical_conditions) or "None recorded"
        social_text = "\n".join(f"- {s}" for s in social_conditions) or "None recorded"

    if isinstance(encounters, DataUnavailable):
        encounter_count = None
        encounters_text = str(encounters)
    else:
        encounter_count = len(encounters)
        encounters_text = f"{encounter_count} in the last 5 on record"

    medications = summary["current_medications"]
    medications_text = (
        summary["data_unavailable"]["current_medications"]
        if medications is None
        else f"{len(medications)} on record"
    )

    lab_summary = []
    for obs in ([] if isinstance(observations, DataUnavailable) else observations[:8]):
        name = obs.get("code", {}).get("text", "")
        value = obs.get("valueQuantity", {})
        val_str = f"{value.get('value', '?')} {value.get('unit', '')}".strip()
        interp = (
            obs.get("interpretation", [{}])[0]
            .get("coding", [{}])[0]
            .get("code", "")
        )
        flag = " [ABNORMAL]" if interp in ("H", "L", "HH", "LL", "A") else ""
        if name:
            lab_summary.append(f"- {name}: {val_str}{flag}")

    if isinstance(observations, DataUnavailable):
        lab_text = str(observations)
    else:
        lab_text = "\n".join(lab_summary) or "No lab data available"

    prompt = f"""
You are a clinical risk analyst writing a readmission risk assessment.
Use only the data provided. Do not invent clinical details.{ACCESS_DENIED_RULE.rstrip() if notes else ""}

PATIENT: {summary['name']} | DOB: {summary['birth_date']}
RECENT ENCOUNTERS: {encounters_text}
ACTIVE MEDICATIONS: {medications_text}

CLINICAL CONDITIONS — these determine the risk level:
{clinical_text}

RECENT LAB VALUES:
{lab_text}

PSYCHOSOCIAL AND SOCIAL HISTORY — contextual barriers only, not primary risk drivers:
{social_text}

Respond in exactly this format:

RISK LEVEL: [LOW / MODERATE / HIGH]

KEY CLINICAL RISK DRIVERS:
(2-4 factors from clinical conditions and labs — not social history)

CONTEXTUAL BARRIERS:
(1-3 social or environmental factors that could complicate recovery)

NARRATIVE:
(2-3 plain English sentences for a care coordinator. Lead with the clinical picture, end with barriers.)

RECOMMENDED ACTIONS:
(1-3 concrete steps to reduce readmission risk)
"""

    narrative_text = await generate(prompt, temperature=0.15)

    risk_level = "UNKNOWN"
    for line in narrative_text.split("\n"):
        if line.startswith("RISK LEVEL:"):
            risk_level = line.replace("RISK LEVEL:", "").strip()
            break

    return {
        "patient_id": patient_id,
        "patient_name": summary["name"],
        "risk_level": risk_level,
        "narrative": narrative_text,
        "encounter_count_last5": encounter_count,
        "data_unavailable": notes,
    }