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
    get_patient_conditions,
    get_patient_encounters,
    get_patient_observations,
)
from llm.groq_client import generate
from tools.patient_summary import get_patient_summary
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

    clinical_conditions, social_conditions = _split_conditions(raw_conditions)
    encounter_count = len(encounters)

    lab_summary = []
    for obs in observations[:8]:
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

    prompt = f"""
You are a clinical risk analyst writing a readmission risk assessment.
Use only the data provided. Do not invent clinical details.

PATIENT: {summary['name']} | DOB: {summary['birth_date']}
RECENT ENCOUNTERS: {encounter_count} in the last 5 on record
ACTIVE MEDICATIONS: {len(summary['current_medications'])} on record

CLINICAL CONDITIONS — these determine the risk level:
{chr(10).join(f"- {c}" for c in clinical_conditions) or "None recorded"}

RECENT LAB VALUES:
{chr(10).join(lab_summary) or "No lab data available"}

PSYCHOSOCIAL AND SOCIAL HISTORY — contextual barriers only, not primary risk drivers:
{chr(10).join(f"- {s}" for s in social_conditions) or "None recorded"}

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
    }