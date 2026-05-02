"""
Tool: generate_transition_brief
--------------------------------
The core tool. Pulls patient FHIR context from Po and uses
Groq (LLaMA 3.3 70B) to generate a structured handoff document.
"""

from typing import Optional
from fhir.client import get_patient_encounters, get_patient_observations
from llm.groq_client import generate
from tools.patient_summary import get_patient_summary


async def generate_transition_brief(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
    transition_type: str = "hospital-to-home",
) -> dict:
    summary = await get_patient_summary(fhir_url, patient_id, fhir_token)
    encounters = await get_patient_encounters(fhir_url, patient_id, fhir_token, limit=3)
    observations = await get_patient_observations(fhir_url, patient_id, fhir_token, category="laboratory")

    encounter_text = ""
    for enc in encounters:
        enc_type = enc.get("type", [{}])[0].get("text", "Visit")
        enc_date = enc.get("period", {}).get("start", "Unknown date")
        reason = enc.get("reasonCode", [{}])[0].get("text", "No reason recorded")
        encounter_text += f"- {enc_type} on {enc_date}: {reason}\n"

    lab_text = ""
    for obs in observations[:5]:
        name = obs.get("code", {}).get("text", "Lab")
        value = obs.get("valueQuantity", {})
        val_str = f"{value.get('value', '?')} {value.get('unit', '')}".strip()
        date = obs.get("effectiveDateTime", "Unknown date")
        lab_text += f"- {name}: {val_str} ({date})\n"

    prompt = f"""
You are a clinical documentation assistant. Generate a concise, structured care transition brief.
Use only the information provided. Do not infer or add clinical details not present in the data.

PATIENT: {summary['name']} | DOB: {summary['birth_date']}
TRANSITION TYPE: {transition_type}

ACTIVE CONDITIONS:
{chr(10).join(f"- {c}" for c in summary['active_conditions']) or "None recorded"}

CURRENT MEDICATIONS:
{chr(10).join(f"- {m['medication']}: {m['dosage']}" for m in summary['current_medications']) or "None recorded"}

RECENT ENCOUNTERS:
{encounter_text or "No recent encounters recorded"}

RECENT LAB RESULTS:
{lab_text or "No recent labs recorded"}

Generate a transition brief with these sections:
1. REASON FOR TRANSITION
2. CLINICAL SUMMARY
3. ACTIVE PROBLEMS
4. MEDICATIONS AT DISCHARGE
5. PENDING RESULTS OR ACTIONS
6. FOLLOW-UP REQUIRED

Keep it factual, clinical, and under 400 words.
"""

    brief_text = await generate(prompt, temperature=0.1)

    return {
        "patient_id": patient_id,
        "patient_name": summary["name"],
        "transition_type": transition_type,
        "brief": brief_text,
    }
