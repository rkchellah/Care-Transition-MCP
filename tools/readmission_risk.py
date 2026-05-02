"""
Tool: readmission_risk_narrative
----------------------------------
Reads the full FHIR picture from Po and generates a
plain-language risk narrative. This is where the AI factor
is clearest — no rule-based system synthesizes this.
"""

from typing import Optional
from fhir.client import get_patient_encounters, get_patient_observations
from llm.groq_client import generate
from tools.patient_summary import get_patient_summary


async def readmission_risk_narrative(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
) -> dict:
    summary = await get_patient_summary(fhir_url, patient_id, fhir_token)
    encounters = await get_patient_encounters(fhir_url, patient_id, fhir_token, limit=5)
    observations = await get_patient_observations(fhir_url, patient_id, fhir_token)

    encounter_count = len(encounters)

    lab_summary = []
    for obs in observations[:8]:
        name = obs.get("code", {}).get("text", "")
        value = obs.get("valueQuantity", {})
        val_str = f"{value.get('value', '?')} {value.get('unit', '')}".strip()
        interp = obs.get("interpretation", [{}])[0].get("coding", [{}])[0].get("code", "")
        flag = " [ABNORMAL]" if interp in ("H", "L", "HH", "LL", "A") else ""
        if name:
            lab_summary.append(f"- {name}: {val_str}{flag}")

    prompt = f"""
You are a clinical risk analyst. Write a concise readmission risk narrative.
Use only the data provided. Do not invent clinical details.

PATIENT: {summary['name']} | DOB: {summary['birth_date']}
ACTIVE CONDITIONS: {', '.join(summary['active_conditions']) or 'None recorded'}
CURRENT MEDICATIONS: {len(summary['current_medications'])} active medications
RECENT ENCOUNTERS (last 5): {encounter_count} encounters on record

RECENT LAB VALUES:
{chr(10).join(lab_summary) or "No lab data available"}

Respond in exactly this format:

RISK LEVEL: [LOW / MODERATE / HIGH]

KEY RISK DRIVERS:
(list 2-4 specific factors from the data)

NARRATIVE:
(2-3 sentences in plain English for a non-specialist care coordinator)

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
