"""
FHIR Client
-----------
Reads patient data from the Po workspace FHIR server.

SHARP Context:
    Po passes three values to our MCP server on every tool call:
    - fhir_url:   The workspace FHIR base URL
    - patient_id: The FHIR ID of the selected patient
    - fhir_token: Bearer token for authenticated FHIR access

    These are injected by Po — we don't generate them.

Known issue (April 2026):
    Po is occasionally sending an empty fhir_token, causing 403 errors.

Access failures (401/403):
    Search functions return a DataUnavailable marker instead of an empty list.
    An inaccessible list is not an empty list — the patient may have conditions
    or medications we could not read — so callers must report it, never treat
    it as "none recorded".

All data is synthetic or de-identified. No real PHI.
"""

import httpx
from typing import Optional


class DataUnavailable:
    """Returned by a search when the FHIR server refuses access (401/403)."""

    def __init__(self, label: str, status_code: int):
        self.label = label
        self.status_code = status_code

    def __str__(self) -> str:
        return f"{self.label}: ACCESS DENIED (HTTP {self.status_code}) - verify manually"

    def __repr__(self) -> str:
        return f"DataUnavailable({self.label!r}, {self.status_code})"


def _build_headers(fhir_token: Optional[str]) -> dict:
    """Build FHIR request headers. Include auth token only if present."""
    headers = {"Accept": "application/fhir+json"}
    if fhir_token:
        headers["Authorization"] = f"Bearer {fhir_token}"
    return headers


async def _search(
    fhir_url: str,
    resource: str,
    params: dict,
    fhir_token: Optional[str],
    label: str,
) -> list[dict] | DataUnavailable:
    """Run a FHIR search and return the matching resources (first page only)."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{fhir_url}/{resource}",
            params=params,
            headers=_build_headers(fhir_token),
        )
    if response.status_code in (401, 403):
        # Missing scope, or platform bug: fhir_token occasionally empty from Po
        return DataUnavailable(label, response.status_code)
    response.raise_for_status()
    return [e["resource"] for e in response.json().get("entry", [])]


async def get_patient(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None
) -> dict:
    """Fetch a patient resource by ID from the Po FHIR server."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{fhir_url}/Patient/{patient_id}",
            headers=_build_headers(fhir_token),
        )
        response.raise_for_status()
        return response.json()


async def get_patient_conditions(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None
) -> list[dict] | DataUnavailable:
    """Fetch all active conditions for a patient."""
    return await _search(
        fhir_url,
        "Condition",
        {"patient": patient_id, "clinical-status": "active"},
        fhir_token,
        "Conditions",
    )


async def get_patient_medications(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None
) -> list[dict] | DataUnavailable:
    """Fetch current medication requests for a patient."""
    return await _search(
        fhir_url,
        "MedicationRequest",
        {"patient": patient_id, "status": "active"},
        fhir_token,
        "Medications",
    )


async def get_patient_encounters(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None, limit: int = 5
) -> list[dict] | DataUnavailable:
    """Fetch recent encounters for a patient."""
    return await _search(
        fhir_url,
        "Encounter",
        {"patient": patient_id, "_sort": "-date", "_count": limit},
        fhir_token,
        "Encounters",
    )


async def get_patient_observations(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
    category: str = "laboratory",
) -> list[dict] | DataUnavailable:
    """Fetch recent lab results or vitals for a patient."""
    return await _search(
        fhir_url,
        "Observation",
        {"patient": patient_id, "category": category, "_sort": "-date", "_count": 10},
        fhir_token,
        "Vital signs" if category == "vital-signs" else "Lab results",
    )


async def get_patient_appointments(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None
) -> list[dict] | DataUnavailable:
    """Fetch upcoming appointments for a patient."""
    return await _search(
        fhir_url,
        "Appointment",
        {"patient": patient_id, "status": "booked"},
        fhir_token,
        "Appointments",
    )
