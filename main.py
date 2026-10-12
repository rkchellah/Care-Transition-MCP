"""
Care Transition MCP Server
--------------------------
Exposes 5 tools for safe patient care transitions using Po FHIR data and Groq (LLaMA 3.3 70B).

FHIR Context:
    The middleware below copies the FHIR headers into a contextvar on every
    HTTP request. With SSE in mcp 1.9.0, a POST to /messages/ only queues the
    tool call; the tool runs in a task spawned from the long-lived GET /sse
    connection, so it reads the headers set when that connection was opened.

    This works today because Po sends the headers on the initial GET /sse.
    Known limitation: if the user switches patient (or the token changes)
    without reconnecting, tools could use the previous patient's ID and token.
    Fixing it needs an mcp upgrade (POST request visible to tools) - deferred.

    Headers captured:
    - X-FHIR-Server-URL
    - X-FHIR-Access-Token
    - X-Patient-ID
"""

import os
import uvicorn
from contextvars import ContextVar

from mcp.server.fastmcp import FastMCP, Context
from tools.patient_summary import get_patient_summary
from tools.transition_brief import generate_transition_brief
from tools.followup_gaps import flag_followup_gaps
from tools.medication_reconciliation import medication_reconciliation
from tools.readmission_risk import readmission_risk_narrative

# Context variable — stores FHIR headers for the current request
# Defaults to empty dict so tools never crash on missing headers
_fhir_ctx: ContextVar[dict] = ContextVar("fhir_ctx", default={})

# Cloud Run sets PORT; default to 8080 locally
PORT = int(os.getenv("PORT", "8080"))

mcp = FastMCP(
    "care-transition-mcp",
    host="0.0.0.0",
    port=PORT,
)

# --- SHARP / Po FHIR Extension declaration ---
_original_init_options = mcp._mcp_server.create_initialization_options

FHIR_SCOPES = [
    {"name": "patient/Patient.rs", "required": True},
    {"name": "patient/Condition.rs"},
    {"name": "patient/MedicationRequest.rs"},
    {"name": "patient/Encounter.rs"},
    {"name": "patient/Observation.rs"},
    {"name": "patient/Appointment.rs"},
]

def _patched_init_options(*args, **kwargs):
    opts = _original_init_options(*args, **kwargs)
    if opts.capabilities.experimental is None:
        opts.capabilities.experimental = {}
    opts.capabilities.experimental["fhir_context_required"] = {"value": True}
    if opts.capabilities.__pydantic_extra__ is None:
        opts.capabilities.__pydantic_extra__ = {}
    opts.capabilities.__pydantic_extra__["extensions"] = {
        "ai.promptopinion/fhir-context": {"scopes": FHIR_SCOPES}
    }
    return opts

mcp._mcp_server.create_initialization_options = _patched_init_options
# --- end FHIR extension ---


class FhirHeaderMiddleware:
    """
    Raw ASGI middleware that captures FHIR context headers from every
    incoming HTTP request and stores them in a contextvar.

    Tools see the value set on the GET /sse connection, not on the POST
    that carries the tool call (see module docstring).

    Must be raw ASGI (not BaseHTTPMiddleware) because BaseHTTPMiddleware
    is incompatible with SSE streaming responses — it buffers the body
    and crashes with: AssertionError: Unexpected message http.response.start

    Bug fix: Replaced BaseHTTPMiddleware with raw ASGI class.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Headers in ASGI scope are list of (name_bytes, value_bytes)
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            token = _fhir_ctx.set({
                "fhir_url": headers.get(b"x-fhir-server-url", b"").decode(),
                "patient_id": headers.get(b"x-patient-id", b"").decode(),
                "fhir_token": headers.get(b"x-fhir-access-token", b"").decode(),
            })
            try:
                await self.app(scope, receive, send)
            finally:
                _fhir_ctx.reset(token)
        else:
            await self.app(scope, receive, send)


def _get_fhir_context() -> tuple[str, str, str]:
    """Read FHIR context from the current request's contextvar."""
    ctx = _fhir_ctx.get()
    return ctx.get("fhir_url", ""), ctx.get("patient_id", ""), ctx.get("fhir_token", "")


@mcp.tool()
async def get_patient_summary_tool(ctx: Context) -> dict:
    """
    Get a structured summary of a patient's clinical profile.
    Returns name, date of birth, active conditions, and current medications.
    Patient context is injected automatically by Po when a patient is selected.
    """
    fhir_url, patient_id, fhir_token = _get_fhir_context()
    if not fhir_url or not patient_id:
        return {"error": "No patient selected. Please select a patient in Po first."}
    return await get_patient_summary(fhir_url=fhir_url, patient_id=patient_id, fhir_token=fhir_token or None)


@mcp.tool()
async def generate_transition_brief_tool(transition_type: str, ctx: Context) -> dict:
    """
    Generate a structured care transition brief for a patient being discharged.
    Uses FHIR data and Groq (LLaMA 3.3 70B) to produce a clinician-ready handoff document.

    Args:
        transition_type: Type of transition — hospital-to-home, ICU-to-ward,
                         hospital-to-SNF, ED-to-admission
    """
    fhir_url, patient_id, fhir_token = _get_fhir_context()
    if not fhir_url or not patient_id:
        return {"error": "No patient selected. Please select a patient in Po first."}
    return await generate_transition_brief(fhir_url=fhir_url, patient_id=patient_id, fhir_token=fhir_token or None, transition_type=transition_type)


@mcp.tool()
async def flag_followup_gaps_tool(ctx: Context) -> dict:
    """
    Identify follow-up care gaps before or after a patient transition.
    Flags missing appointments, abnormal labs, and medication follow-up needs.
    Patient context is injected automatically by Po when a patient is selected.
    """
    fhir_url, patient_id, fhir_token = _get_fhir_context()
    if not fhir_url or not patient_id:
        return {"error": "No patient selected. Please select a patient in Po first."}
    return await flag_followup_gaps(fhir_url=fhir_url, patient_id=patient_id, fhir_token=fhir_token or None)


@mcp.tool()
async def medication_reconciliation_tool(ctx: Context) -> dict:
    """
    Check for medication discrepancies in a patient's current medication list.
    Flags duplicates, missing medications for known conditions, and dosage concerns.
    Patient context is injected automatically by Po when a patient is selected.
    """
    fhir_url, patient_id, fhir_token = _get_fhir_context()
    if not fhir_url or not patient_id:
        return {"error": "No patient selected. Please select a patient in Po first."}
    return await medication_reconciliation(fhir_url=fhir_url, patient_id=patient_id, fhir_token=fhir_token or None)


@mcp.tool()
async def readmission_risk_narrative_tool(ctx: Context) -> dict:
    """
    Generate a plain-language readmission risk narrative for a patient.
    Returns a risk level (LOW/MODERATE/HIGH), key risk drivers, and
    recommended actions for the care team.
    Patient context is injected automatically by Po when a patient is selected.
    """
    fhir_url, patient_id, fhir_token = _get_fhir_context()
    if not fhir_url or not patient_id:
        return {"error": "No patient selected. Please select a patient in Po first."}
    return await readmission_risk_narrative(fhir_url=fhir_url, patient_id=patient_id, fhir_token=fhir_token or None)


if __name__ == "__main__":
    app = mcp.sse_app()
    app.add_middleware(FhirHeaderMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
