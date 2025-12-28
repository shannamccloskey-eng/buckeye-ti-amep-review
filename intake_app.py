import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-5.1"  # OpenAI API model ID


# ---------------- PDF UTILITIES ----------------

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.
    """
    reader = PdfReader(pdf_path)
    all_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        all_text.append(page_text)
    return "\n\n".join(all_text)


# ---------------- OPENAI HELPERS ----------------

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def analyze_plans_for_food_service(
    client: OpenAI,
    full_pdf_text: str,
    project_type: str,
) -> Dict[str, Any]:
    """
    Use GPT to scan the full plan text for food-service signals and specific items.
    """
    user_prompt = f"""
You are assisting the City of Buckeye Building Department with COMMERCIAL PLAN INTAKE
for a project type: "{project_type}".

The uploaded PDF text below is the full plan set (architectural + MEP, etc.).
Your tasks:

1) Determine whether the plans clearly indicate a FOOD SERVICE use
   (restaurant, bar, commercial kitchen, coffee shop, bakery, etc.).
2) For food-service related equipment, indicate whether the plans show or
   reference ANY of the following:
   - grease interceptor (interceptor, grease trap, grease waste, etc.)
   - walk-in cooler
   - walk-in freezer
   - cooking hood (Type I or II, exhaust hood, kitchen hood, etc.)
   - IECC / energy-code compliance details (lighting, envelope, mechanical).

Return ONLY valid JSON in this exact format (no commentary):

{{
  "is_food_service_by_plans": true/false,
  "found_items": {{
    "grease_interceptor": true/false,
    "walk_in_cooler": true/false,
    "walk_in_freezer": true/false,
    "hood": true/false,
    "iecc_energy_code": true/false
  }},
  "notes": "short explanation of what you saw in the plans"
}}

PDF_PLAN_TEXT_START
{full_pdf_text[:120000]}
PDF_PLAN_TEXT_END
"""

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=(
            "You are assisting City of Buckeye intake staff. "
            "Follow the user instructions carefully and respond with STRICT JSON only."
        ),
        input=user_prompt,
    )

    raw_text = response.output_text.strip()

    try:
        data = json.loads(raw_text)
    except Exception:
        data = {
            "is_food_service_by_plans": False,
            "found_items": {
                "grease_interceptor": False,
                "walk_in_cooler": False,
                "walk_in_freezer": False,
                "hood": False,
                "iecc_energy_code": False,
            },
            "notes": "Model response could not be parsed as JSON. Treating as no food-service items detected.",
        }

    found = data.get("found_items", {}) or {}
    defaults = {
        "grease_interceptor": False,
        "walk_in_cooler": False,
        "walk_in_freezer": False,
        "hood": False,
        "iecc_energy_code": False,
    }
    merged_found = {k: bool(found.get(k, v)) for k, v in defaults.items()}

    return {
        "is_food_service_by_plans": bool(data.get("is_food_service_by_plans", False)),
        "found_items": merged_found,
        "notes": data.get("notes", ""),
    }


def classify_support_doc(
    client: OpenAI,
    doc_text: str,
    filename: str,
) -> Dict[str, Any]:
    """
    Classify a single supporting document into the known document types.
    """
    user_prompt = f"""
You are assisting City of Buckeye Building Department with COMMERCIAL PLAN INTAKE.

A single supporting document has been uploaded with filename: "{filename}".

Based on the text content provided, classify whether this document is any of
the following types (multiple may be true):

- geotech_report: geotechnical / soils report
- deferred_truss_form: deferred truss submittal form or truss design summary
- special_inspections_form: special inspections statement, schedule, or agreement
- struct_calcs: structural calculations (beams, columns, foundations, etc.)
- struct_sheets: separate structural sheets or structural framing plans
- grease_interceptor: grease interceptor details, sizing, or shop drawings
- walk_in_cooler: walk-in cooler submittal, cut sheet, or layout
- walk_in_freezer: walk-in freezer submittal, cut sheet, or layout
- hood_specs: hood specifications, hood schedule, or hood shop drawings
- iecc_docs: IECC / energy code compliance forms, COMcheck, or similar

Return ONLY valid JSON in this exact format (no commentary):

{{
  "doc_types": {{
    "geotech_report": true/false,
    "deferred_truss_form": true/false,
    "special_inspections_form": true/false,
    "struct_calcs": true/false,
    "struct_sheets": true/false,
    "grease_interceptor": true/false,
    "walk_in_cooler": true/false,
    "walk_in_freezer": true/false,
    "hood_specs": true/false,
    "iecc_docs": true/false
  }},
  "notes": "short explanation of your classification"
}}

DOC_TEXT_START
{doc_text[:40000]}
DOC_TEXT_END
"""

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=(
            "You are assisting City of Buckeye intake staff. "
            "Follow the user instructions carefully and respond with STRICT JSON only."
        ),
        input=user_prompt,
    )

    raw_text = response.output_text.strip()
    try:
        data = json.loads(raw_text)
    except Exception:
        data = {
            "doc_types": {},
            "notes": "Could not parse model output as JSON.",
        }

    defaults = {
        "geotech_report": False,
        "deferred_truss_form": False,
        "special_inspections_form": False,
        "struct_calcs": False,
        "struct_sheets": False,
        "grease_interceptor": False,
        "walk_in_cooler": False,
        "walk_in_freezer": False,
        "hood_specs": False,
        "iecc_docs": False,
    }
    found_types = data.get("doc_types", {}) or {}
    merged = {k: bool(found_types.get(k, v)) for k, v in defaults.items()}

    return {
        "doc_types": merged,
        "notes": data.get("notes", ""),
    }


# ---------------- INTAKE LOGIC ----------------

DOC_LABELS = {
    "completed_application": "Completed Permit Application",
    "arch_set": "Architectural Plan Set",
    "mep_set": "MEP Plan Set (Mechanical, Electrical, Plumbing)",
    "geotech_report": "Geotechnical Report",
    "deferred_truss_form": "Deferred Truss Submittal Form",
    "special_inspections_form": "Special Inspections Form",
    "struct_calcs": "Structural Calculations",
    "struct_sheets": "Structural Sheets",
    "grease_interceptor": "Grease Interceptor Details/Submittal",
    "walk_in_cooler": "Walk-in Cooler Submittal/Details",
    "walk_in_freezer": "Walk-in Freezer Submittal/Details",
    "hood_specs": "Hood Specifications / Hood Schedule",
    "iecc_docs": "IECC / Energy Code Compliance Documentation",
}


def build_intake_checklist(
    project_type: str,
    is_food_service_checkbox: bool,
    plan_analysis: Dict[str, Any],
    ai_detected_docs: Dict[str, bool],
    doc_sources: Dict[str, List[str]],
) -> Dict[str, Any]:
    rows = []

    def add_row(code: str, required: bool, note: str = ""):
        label = DOC_LABELS.get(code, code)
        provided = bool(ai_detected_docs.get(code, False))
        if required and not provided:
            status = "MISSING – Required for intake"
        elif required and provided:
            status = "Provided – Required"
        elif (not required) and provided:
            status = "Provided – Not strictly required"
        else:
            status = "Not required"

        src_files = doc_sources.get(code, [])
        src_note = ""
        if src_files:
            src_note = f"Provided in file(s): {', '.join(src_files)}"
            if note:
                note = note + " " + src_note
            else:
                note = src_note

        rows.append(
            {
                "code": code,
                "document": label,
                "required": required,
                "provided": provided,
                "status": status,
                "notes": note,
            }
        )

    # Always-required docs
    add_row("completed_application", required=True, note="Verified by intake staff from application submittal.")
    add_row("arch_set", required=True, note="Typically included within the main plan PDF.")
    add_row("mep_set", required=True, note="Typically included within the main plan PDF.")

    # Shell / ground-up extras
    if project_type in ("Shell Building", "Ground-Up"):
        add_row("geotech_report", True)
        add_row("deferred_truss_form", True)
        add_row("special_inspections_form", True)
        add_row("struct_calcs", True)
        add_row("struct_sheets", True)

    # Food-service-driven docs
    analysis_food = bool(plan_analysis.get("is_food_service_by_plans", False))
    found_food_items = plan_analysis.get("found_items", {}) or {}
    any_food_feature = any(found_food_items.values())

    is_food_context = (
        (project_type in ("Ground-Up", "Tenant Improvement (TI)"))
        and (is_food_service_checkbox or analysis_food or any_food_feature)
    )

    if is_food_context:
        base_note = "Required if plans show this food-service equipment or system."
        add_row(
            "grease_interceptor",
            required=bool(found_food_items.get("grease_interceptor", False)),
            note=base_note,
        )
        add_row(
            "walk_in_cooler",
            required=bool(found_food_items.get("walk_in_cooler", False)),
            note=base_note,
        )
        add_row(
            "walk_in_freezer",
            required=bool(found_food_items.get("walk_in_freezer", False)),
            note=base_note,
        )
        add_row(
            "hood_specs",
            required=bool(found_food_items.get("hood", False)),
            note=base_note,
        )
        add_row(
            "iecc_docs",
            required=bool(found_food_items.get("iecc_energy_code", False)),
            note="Required when IECC-specific details or energy compliance are called out.",
        )

    missing_required = [r["document"] for r in rows if r["required"] and not r["provided"]]

    if missing_required:
        intake_status = "NOT READY FOR INTAKE – Missing required documents."
    else:
        intake_status = "READY FOR INTAKE – All required documents submitted."

    return {
        "rows": rows,
        "missing_required": missing_required,
        "intake_status": intake_status,
        "plan_analysis": plan_analysis,
    }


# ---------------- STREAMLIT UI ----------------

def main(embed: bool = False):
    """
    Commercial Plan Intake UI.

    embed = False → standalone (run this file directly).
    embed = True  → called from master tabbed app.
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – Commercial Plan Intake (Beta)",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – Commercial Plan Intake (Build 1.0)")
    st.caption(f"Model: `{MODEL_NAME}`")

    st.write(
        "This tool helps determine whether a COMMERCIAL plan submittal is complete for intake. "
        "It checks required documents based on project type and whether the plans show food-service features."
    )

    st.info(
        "Step 1: Choose project type and indicate if it is food service.\n"
        "Step 2: Upload the full plan set PDF.\n"
        "Step 3: Upload ALL supporting documents you believe are required "
        "(geotech report, special inspections form, hood specs, etc.).\n"
        "Step 4: Run the intake check to verify completeness."
    )

    # ---- API key (no sidebar) ----
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the app."
        )
        st.stop()

    client = get_client(env_api_key)

    # --- Main form ---

    st.subheader("Project Information")

    project_type = st.selectbox(
        "Project Type",
        ["Shell Building", "Ground-Up", "Tenant Improvement (TI)"],
    )

    is_food_service_checkbox = False
    if project_type in ("Ground-Up", "Tenant Improvement (TI)"):
        is_food_service_checkbox = st.checkbox(
            "This project is a FOOD SERVICE use (restaurant, bar, commercial kitchen, café, etc.)",
            value=False,
        )

    st.subheader("Plan Set")

    plan_pdf = st.file_uploader(
        "Upload full plan set PDF (architectural + MEP, etc.)",
        type=["pdf"],
    )

    st.subheader("Supporting Documents")

    st.write(
        "Upload ALL other supporting PDF documents here (geotechnical report, "
        "special inspections form, structural calcs, hood specs, IECC docs, etc.)."
    )

    support_pdfs = st.file_uploader(
        "Supporting documents (PDF, multiple allowed)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.markdown("---")

    run_button = st.button("Run Intake Check", type="primary")

    if run_button:
        if not plan_pdf:
            st.error("Please upload the full plan set PDF before running the intake check.")
            st.stop()

        if not support_pdfs:
            st.warning(
                "No supporting documents uploaded. The tool will still analyze the plans, "
                "but most required documents will show as missing."
            )

        with st.spinner("Analyzing plans and supporting documents for intake completeness..."):
            # --- Extract text from main plan PDF ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(plan_pdf.read())
                plan_tmp_path = tmp.name

            try:
                plan_text = extract_pdf_text(plan_tmp_path)
            except Exception as e:
                st.error(f"Error extracting text from plan PDF: {e}")
                try:
                    os.remove(plan_tmp_path)
                except OSError:
                    pass
                return
            finally:
                try:
                    os.remove(plan_tmp_path)
                except OSError:
                    pass

            # --- Analyze plans for food-service features ---
            try:
                plan_analysis = analyze_plans_for_food_service(
                    client, plan_text, project_type
                )
            except Exception as e:
                st.error(f"Error analyzing plans with AI: {e}")
                return

            # --- Analyze supporting docs and aggregate detected document types ---
            ai_detected_docs: Dict[str, bool] = {}
            doc_sources: Dict[str, List[str]] = {}

            if support_pdfs:
                for upload in support_pdfs:
                    filename = upload.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(upload.read())
                        tmp_path = tmp.name

                    try:
                        doc_text = extract_pdf_text(tmp_path)
                    except Exception:
                        doc_text = ""
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                    if not doc_text.strip():
                        continue

                    try:
                        classification = classify_support_doc(client, doc_text, filename)
                    except Exception:
                        continue

                    doc_types = classification.get("doc_types", {}) or {}
                    for code, is_true in doc_types.items():
                        if is_true:
                            ai_detected_docs[code] = True
                            doc_sources.setdefault(code, []).append(filename)

            # Assume main plan includes these:
            ai_detected_docs.setdefault("completed_application", True)
            ai_detected_docs.setdefault("arch_set", True)
            ai_detected_docs.setdefault("mep_set", True)

            checklist = build_intake_checklist(
                project_type=project_type,
                is_food_service_checkbox=is_food_service_checkbox,
                plan_analysis=plan_analysis,
                ai_detected_docs=ai_detected_docs,
                doc_sources=doc_sources,
            )

        # --- Display results ---

        st.subheader("Intake Status")
        st.markdown(f"**{checklist['intake_status']}**")

        if checklist["missing_required"]:
            st.error(
                "Missing required documents:\n- "
                + "\n- ".join(checklist["missing_required"])
            )
        else:
            st.success("All required documents appear to be submitted for this intake.")

        st.subheader("Intake Checklist")

        display_rows = [
            {
                "Document": r["document"],
                "Required?": "Yes" if r["required"] else "No",
                "Provided?": "Yes" if r["provided"] else "No",
                "Status": r["status"],
                "Notes": r["notes"],
            }
            for r in checklist["rows"]
        ]
        st.table(display_rows)

        st.subheader("Plan Analysis (Food-Service Signals from Plan Set)")
        st.json(checklist["plan_analysis"])

        st.caption(
            "This tool performs an intake completeness check only. "
            "It does not constitute a full code compliance review."
        )


if __name__ == "__main__":
    main()
