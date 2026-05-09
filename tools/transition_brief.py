"""
Tool: generate_transition_brief
--------------------------------
The core tool. Pulls patient FHIR context from Po and uses
Groq (LLaMA 3.3 70B) to generate a structured handoff document.

Fix (May 2026):
    Conditions are split into clinical vs social before hitting the LLM.
    This prevents the SDoH Trap — where social history (employment, education)
    gets weighted the same as active medical problems in the output.

    Vitals are fetched separately from labs so the LLM can speak to
    the patient's current physiological state.

    Medication zero-warning added for ICU transitions — no medications
    on an ICU patient is a safety flag, not a clean bill of health.

    Nursing Care Plan section added — mirrors real clinical handover
    structure where the receiving team gets immediate action items,
    not just a data summary.

    Section 3 hardened — social conditions explicitly blocked from
    appearing in Active Medical Problems regardless of LLM reasoning.
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


def _is_icu_transition(transition_type: str) -> bool:
    """Return True if this is an ICU-level transition."""
    icu_keywords = ["icu", "intensive", "critical"]
    return any(k in transition_type.lower() for k in icu_keywords)


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
    for obs in vitals[:5]:
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
    primary_reason = (
        "Not recorded — clinician must verify primary admission diagnosis "
        "before accepting transfer"
    )
    encounter_history = ""
    for enc in encounters:
        reason = enc.get("reasonCode", [{}])[0].get("text", "")
        enc_type = enc.get("type", [{}])[0].get("text", "Visit")
        enc_date = enc.get("period", {}).get("start", "Unknown date")
        if reason and primary_reason.startswith("Not recorded"):
            primary_reason = reason
        encounter_history += (
            f"- {enc_type} on {enc_date}: {reason or 'No reason recorded'}\n"
        )

    # Medication safety language
    # Zero medications on an ICU patient is a red flag, not a clean record
    if summary["current_medications"]:
        medications_text = "\n".join(
            f"- {m['medication']}: {m['dosage']}"
            for m in summary["current_medications"]
        )
        medication_warning = ""
    else:
        medications_text = "No active medications found in FHIR records."
        if _is_icu_transition(transition_type):
            medication_warning = (
                "SAFETY FLAG: It is highly irregular for an ICU patient to have zero "
                "medications on record. The receiving team must verify what was administered "
                "in the ICU — IV fluids, antibiotics, vasopressors, sedation — and confirm "
                "what requires continuation, titration, or discontinuation on the ward. "
                "Do not assume this patient is medication-free."
            )
        else:
            medication_warning = (
                "NOTE: No medications on record. Confirm with the clinical team whether "
                "this reflects a genuine clinical status or a gap in FHIR documentation."
            )

    prompt = f"""
You are a senior clinical nurse writing a formal handover brief for the receiving care team.
Write like a clinician handing over to the next shift — direct, prioritised, action-oriented.
Use only the data provided. Do not add clinical details not in the data.

PATIENT: {summary['name']} | DOB: {summary['birth_date']}
TRANSITION: {transition_type}

---

Write the handover using these exact section headings, in this exact order:

1. PRIMARY CLINICAL IMPRESSION AND REASON FOR TRANSFER
State the admission reason. If not recorded, say so explicitly and flag that the receiving
clinician must verify before accepting the transfer.
Admission reason: {primary_reason}

2. RECENT PHYSIOLOGICAL STATUS
Summarise the patient's current physical state from the vitals below.
List each vital with its value and date.
If any standard vital is missing (Pulse, BP, RR, SpO2, Temp), name it explicitly as
missing and instruct the receiving nurse to obtain it immediately on arrival.

RECENT VITALS:
{vitals_text or "No recent vitals recorded — full set required before transfer."}

3. ACTIVE MEDICAL PROBLEMS
List clinical diagnoses only — conditions such as infections, chronic disease,
organ failure, or physical injury found in the medical problem list.
If the clinical problem list is empty, write exactly this one line:
"No acute clinical conditions flagged in FHIR records."
Do not mention social history, employment, education, abuse history, social contact,
or any psychosocial finding in this section under any circumstances.
Those belong in section 6 only. This section is for medical diagnoses only.
{chr(10).join(f"- {c}" for c in clinical_conditions) or "No acute clinical conditions flagged in FHIR records."}

4. MEDICATION RECONCILIATION
{medications_text}

{medication_warning}

5. RECENT LABS AND PENDING RESULTS
{lab_text or "No recent lab results recorded."}

6. PSYCHOSOCIAL CONTEXT AND DISCHARGE BARRIERS
This section covers social history only — relevant to discharge planning and safety,
not immediate medical stability for the ward transfer.
Write a brief assessment of how these factors affect the transition, not just a list.
{chr(10).join(f"- {s}" for s in social_conditions) or "No social history recorded."}

7. NURSING CARE PLAN — IMMEDIATE ACTION ITEMS
Based on the gaps identified above, list 2-4 concrete actions for the receiving nurse
in priority order. Format each as:
ACTION: [what to do and why]

Address the most critical gaps first — missing vitals, medication verification,
social safety concerns, and follow-up that must happen before ward discharge.

RECENT ENCOUNTER HISTORY (context only, do not repeat in sections above):
{encounter_history or "No recent encounters recorded."}

---

Keep the total brief under 500 words.
Write like a nurse, not a report generator.
Lead with what the next clinician needs to act on, not what looks complete on paper.
"""

    brief_text = await generate(prompt, temperature=0.1)

    return {
        "patient_id": patient_id,
        "patient_name": summary["name"],
        "transition_type": transition_type,
        "brief": brief_text,
    }