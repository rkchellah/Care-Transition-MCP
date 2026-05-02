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
    We handle this gracefully by returning empty lists rather than crashing.

All data is synthetic or de-identified. No real PHI.
"""

import httpx
from typing import Optional


def _build_headers(fhir_token: Optional[str]) -> dict:
    """Build FHIR request headers. Include auth token only if present."""
    headers = {"Accept": "application/fhir+json"}
    if fhir_token:
        headers["Authorization"] = f"Bearer {fhir_token}"
    return headers


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
) -> list[dict]:
    """Fetch all active conditions for a patient."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{fhir_url}/Condition",
                params={"patient": patient_id, "clinical-status": "active"},
                headers=_build_headers(fhir_token),
            )
            response.raise_for_status()
            return [e["resource"] for e in response.json().get("entry", [])]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            # Platform bug: fhir_token occasionally empty from Po
            return []
        raise


async def get_patient_medications(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None
) -> list[dict]:
    """Fetch current medication requests for a patient."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{fhir_url}/MedicationRequest",
                params={"patient": patient_id, "status": "active"},
                headers=_build_headers(fhir_token),
            )
            response.raise_for_status()
            return [e["resource"] for e in response.json().get("entry", [])]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return []
        raise


async def get_patient_encounters(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None, limit: int = 5
) -> list[dict]:
    """Fetch recent encounters for a patient."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{fhir_url}/Encounter",
                params={"patient": patient_id, "_sort": "-date", "_count": limit},
                headers=_build_headers(fhir_token),
            )
            response.raise_for_status()
            return [e["resource"] for e in response.json().get("entry", [])]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return []
        raise


async def get_patient_observations(
    fhir_url: str,
    patient_id: str,
    fhir_token: Optional[str] = None,
    category: str = "laboratory",
) -> list[dict]:
    """Fetch recent lab results or vitals for a patient."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{fhir_url}/Observation",
                params={
                    "patient": patient_id,
                    "category": category,
                    "_sort": "-date",
                    "_count": 10,
                },
                headers=_build_headers(fhir_token),
            )
            response.raise_for_status()
            return [e["resource"] for e in response.json().get("entry", [])]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return []
        raise


async def get_patient_appointments(
    fhir_url: str, patient_id: str, fhir_token: Optional[str] = None
) -> list[dict]:
    """Fetch upcoming appointments for a patient."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{fhir_url}/Appointment",
                params={"patient": patient_id, "status": "booked"},
                headers=_build_headers(fhir_token),
            )
            response.raise_for_status()
            return [e["resource"] for e in response.json().get("entry", [])]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return []
        raise
