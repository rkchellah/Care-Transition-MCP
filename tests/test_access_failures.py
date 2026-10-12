"""
Tests for FHIR access-failure handling.

A 401/403 on a resource must produce an explicit ACCESS DENIED marker, never an
empty list. A genuinely empty bundle must still read as "none recorded".

All HTTP is served by httpx.MockTransport and the Groq client is mocked, so no
network or API quota is used.

Run from the repo root:  python -m unittest discover tests
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from fhir import client as fhir_client
from fhir.client import DataUnavailable
from tools.followup_gaps import flag_followup_gaps
from tools.medication_reconciliation import medication_reconciliation
from tools.patient_summary import get_patient_summary
from tools.readmission_risk import readmission_risk_narrative
from tools.transition_brief import generate_transition_brief

FHIR_URL = "https://fhir.test/api"
PATIENT_ID = "p1"
TOKEN = "test-token"
LLM_OUTPUT = "RISK LEVEL: HIGH\nLLM OUTPUT"

CONDITIONS_DENIED = "Conditions: ACCESS DENIED (HTTP 403) - verify manually"
MEDICATIONS_DENIED = "Medications: ACCESS DENIED (HTTP 403) - verify manually"

PATIENT = {
    "resourceType": "Patient",
    "id": PATIENT_ID,
    "name": [{"given": ["Ana"], "family": "Test"}],
    "birthDate": "1950-01-01",
}


def _cat(code):
    return [{"coding": [{"code": code}]}]


BUNDLES = {
    "Condition": [
        {"code": {"text": "Hypertension"}, "category": _cat("problem-list-item")},
        {"code": {"text": "Full-time employment"}, "category": _cat("social-history")},
    ],
    "MedicationRequest": [
        {
            "status": "active",
            "medicationCodeableConcept": {"text": "Lisinopril 10 MG"},
            "dosageInstruction": [{"text": "1 tablet daily"}],
        }
    ],
    "Encounter": [
        {
            "type": [{"text": "Emergency"}],
            "reasonCode": [{"text": "Chest pain"}],
            "period": {"start": "2026-01-02"},
        }
    ],
    "vital-signs": [
        {
            "code": {"text": "Heart rate"},
            "valueQuantity": {"value": 88, "unit": "/min"},
            "effectiveDateTime": "2026-01-02",
        }
    ],
    "laboratory": [
        {
            "code": {"text": "Glucose"},
            "valueQuantity": {"value": 180, "unit": "mg/dL"},
            "effectiveDateTime": "2026-01-02",
            "interpretation": [{"coding": [{"code": "H"}]}],
        }
    ],
    "Appointment": [
        {"serviceType": [{"text": "Cardiology"}], "start": "2026-02-01", "status": "booked"}
    ],
}


def _key(request: httpx.Request) -> str:
    resource = request.url.path.rsplit("/", 1)[-1]
    if resource == "Observation":
        return request.url.params["category"]
    return resource


class FhirTestCase(unittest.IsolatedAsyncioTestCase):
    """Serves PATIENT plus BUNDLES; `status` overrides the response for chosen keys."""

    bundles = BUNDLES
    status: dict = {}

    def _handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/api/Patient/{PATIENT_ID}":
            return httpx.Response(200, json=PATIENT)
        key = _key(request)
        if key in self.status:
            return httpx.Response(self.status[key], json={"resourceType": "OperationOutcome"})
        entries = [{"resource": r} for r in self.bundles.get(key, [])]
        bundle = {"resourceType": "Bundle", "type": "searchset"}
        if entries:
            bundle["entry"] = entries
        return httpx.Response(200, json=bundle)

    def setUp(self):
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self._handler)

        def mock_client(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        self.llm = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=LLM_OUTPUT))]
            )
        )
        groq = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.llm)))

        for p in (
            patch.object(fhir_client.httpx, "AsyncClient", mock_client),
            patch("llm.groq_client._get_client", return_value=groq),
        ):
            p.start()
            self.addCleanup(p.stop)

    @property
    def prompt(self) -> str:
        return self.llm.call_args.kwargs["messages"][0]["content"]


class AccessDeniedTests(FhirTestCase):
    status = {"Condition": 403, "MedicationRequest": 403}

    async def test_client_returns_marker_not_empty_list(self):
        conditions = await fhir_client.get_patient_conditions(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIsInstance(conditions, DataUnavailable)
        self.assertEqual(str(conditions), CONDITIONS_DENIED)

    async def test_summary_marks_lists_unavailable(self):
        summary = await get_patient_summary(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIsNone(summary["active_conditions"])
        self.assertIsNone(summary["current_medications"])
        self.assertEqual(
            summary["data_unavailable"],
            {"active_conditions": CONDITIONS_DENIED, "current_medications": MEDICATIONS_DENIED},
        )

    async def test_marker_reaches_prompt_and_brief(self):
        result = await generate_transition_brief(FHIR_URL, PATIENT_ID, TOKEN)

        self.assertEqual(result["data_unavailable"], [CONDITIONS_DENIED, MEDICATIONS_DENIED])
        self.assertIn(CONDITIONS_DENIED, self.prompt)
        self.assertIn(MEDICATIONS_DENIED, self.prompt)
        self.assertIn('Any line marked "ACCESS DENIED"', self.prompt)
        # The "empty" wording must not appear as data
        self.assertNotIn("No active medications found in FHIR records.", self.prompt)
        self.assertNotIn("No social history recorded.", self.prompt)
        self.assertNotIn("\nNo acute clinical conditions flagged in FHIR records.\n", self.prompt)
        # The warning is in the brief even if the LLM drops it
        self.assertTrue(result["brief"].startswith("DATA UNAVAILABLE:\n"))
        self.assertIn(CONDITIONS_DENIED, result["brief"])
        self.assertIn(MEDICATIONS_DENIED, result["brief"])
        self.assertTrue(result["brief"].endswith(LLM_OUTPUT))

    async def test_other_tools_report_marker(self):
        for tool in (flag_followup_gaps, medication_reconciliation, readmission_risk_narrative):
            with self.subTest(tool=tool.__name__):
                result = await tool(FHIR_URL, PATIENT_ID, TOKEN)
                self.assertIn(CONDITIONS_DENIED, result["data_unavailable"])
                self.assertIn(MEDICATIONS_DENIED, result["data_unavailable"])
                self.assertIn(CONDITIONS_DENIED, self.prompt)
                self.assertIn(MEDICATIONS_DENIED, self.prompt)

        result = await medication_reconciliation(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIsNone(result["medication_count"])


class UnauthorizedEncounterTests(FhirTestCase):
    status = {"Encounter": 401}

    async def test_401_is_reported_as_access_denied(self):
        result = await generate_transition_brief(FHIR_URL, PATIENT_ID, TOKEN)
        denied = "Encounters: ACCESS DENIED (HTTP 401) - verify manually"
        self.assertEqual(result["data_unavailable"], [denied])
        self.assertIn(f"Admission reason: {denied}", self.prompt)


class ServerErrorTests(FhirTestCase):
    status = {"Condition": 500}

    async def test_other_errors_still_raise(self):
        with self.assertRaises(httpx.HTTPStatusError):
            await fhir_client.get_patient_conditions(FHIR_URL, PATIENT_ID, TOKEN)


class NormalBundleTests(FhirTestCase):
    async def test_summary_formats_as_before(self):
        summary = await get_patient_summary(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertEqual(summary["name"], "Ana Test")
        self.assertEqual(summary["active_conditions"], ["Hypertension", "Full-time employment"])
        self.assertEqual(
            summary["current_medications"],
            [{"medication": "Lisinopril 10 MG", "dosage": "1 tablet daily"}],
        )
        self.assertEqual(summary["data_unavailable"], {})

    async def test_brief_formats_as_before(self):
        result = await generate_transition_brief(FHIR_URL, PATIENT_ID, TOKEN)

        self.assertEqual(result["brief"], LLM_OUTPUT)
        self.assertEqual(result["data_unavailable"], [])
        self.assertIn("Admission reason: Chest pain", self.prompt)
        self.assertIn("- Heart rate: 88 /min (2026-01-02)", self.prompt)
        self.assertIn("Those belong in section 6 only. This section is for medical diagnoses only.\n- Hypertension\n", self.prompt)
        self.assertIn("4. MEDICATION RECONCILIATION\n- Lisinopril 10 MG: 1 tablet daily\n", self.prompt)
        self.assertIn("- Glucose: 180 mg/dL (2026-01-02) [ABNORMAL]", self.prompt)
        self.assertIn("not just a list.\n- Full-time employment\n", self.prompt)
        self.assertNotIn("ACCESS DENIED", self.prompt)

    async def test_other_tools_format_as_before(self):
        result = await flag_followup_gaps(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIn("ACTIVE CONDITIONS: Hypertension, Full-time employment", self.prompt)
        self.assertIn("CURRENT MEDICATIONS: Lisinopril 10 MG", self.prompt)
        self.assertEqual((result["abnormal_lab_count"], result["upcoming_appointments"]), (1, 1))

        result = await readmission_risk_narrative(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIn("RECENT ENCOUNTERS: 1 in the last 5 on record", self.prompt)
        self.assertIn("ACTIVE MEDICATIONS: 1 on record", self.prompt)
        self.assertEqual(result["risk_level"], "HIGH")

        result = await medication_reconciliation(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIn("- Lisinopril 10 MG | Dosage: 1 tablet daily | Status: active", self.prompt)
        self.assertEqual(result["medication_count"], 1)


class EmptyBundleTests(FhirTestCase):
    bundles = {}

    async def test_empty_is_none_recorded_not_access_denied(self):
        summary = await get_patient_summary(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertEqual(summary["active_conditions"], [])
        self.assertEqual(summary["current_medications"], [])
        self.assertEqual(summary["data_unavailable"], {})

        result = await generate_transition_brief(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertEqual(result["data_unavailable"], [])
        self.assertEqual(result["brief"], LLM_OUTPUT)
        self.assertIn("for medical diagnoses only.\nNo acute clinical conditions flagged in FHIR records.\n", self.prompt)
        self.assertIn("No active medications found in FHIR records.", self.prompt)
        self.assertIn("No social history recorded.", self.prompt)
        self.assertNotIn("ACCESS DENIED", self.prompt)

        await readmission_risk_narrative(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIn("these determine the risk level:\nNone recorded\n", self.prompt)
        self.assertNotIn("ACCESS DENIED", self.prompt)

        await flag_followup_gaps(FHIR_URL, PATIENT_ID, TOKEN)
        self.assertIn("ACTIVE CONDITIONS: None\nCURRENT MEDICATIONS: None", self.prompt)
        self.assertNotIn("ACCESS DENIED", self.prompt)


if __name__ == "__main__":
    unittest.main()
