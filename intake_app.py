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
    Ask the model to analyze the plan text for:
      - Is this a food service use?
      - Are there any of the following shown or referenced:
        grease interceptor, walk-in coolers/freezers, hoods, IECC/energy code.
    Returns a dict with:
      {
        "is_food_service_by_plans": bool,
        "found_items": {
          "grease_interceptor": bool,
          "walk_in_cooler": bool,
          "walk_in_freezer": bool,
          "hood": bool,
          "iecc_energy_code": bool
        },
        "notes": "string explanation"
      }
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

    # Try to parse JSON. If it fails, return safe defaults.
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

    # Ensure keys exist and are well-formed
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


# ---------------- INTAKE LOGIC ----------------

def build_intake_checklist(
    project_type: str,
    is_food_service_checkbox: bool,
    plan_analysis: Dict[str, Any],
    docs_provided: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Build an intake checklist given:
      - project_type: "Shell Building", "Ground-Up", "Tenant Improvement (TI)"
      - is_food_service_checkbox: user-indicated food service
      - plan_analysis: result of analyze_plans_for_food_service
      - docs_provided: dict of document_code -> bool (submitted or not)

    Returns dict with:
      {
        "rows": [ { "document": str, "required": bool, "provided": bool, "status": str, "notes": str }, ... ],
        "missing_required": [list of document names],
        "intake_status": "Ready for intake" | "Not ready for intake"
      }
    """
    rows = []

    # Helper to add row
    def add_doc(code: str, label: str, required: bool, note: str = ""):
        provided = bool(docs_provided.get(code, False))
        if required and not provided:
            status = "MISSING – Required for intake"
        elif required and provided:
            status = "Provided – Required"
        elif (not required) and provided:
            status = "Provided – Not strictly required"
        else:
            status = "Not required"
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

    # 1) Documents required for ALL project types
    add_doc("completed_application", "Completed Permit Application", True)
    add_doc("arch_set", "Architectural Plan Set", True)
    add_doc("mep_set", "MEP Plan Set (Mechanical, Electrical, Plumbing)", True)

    # 2) Extra requirements for Shell + Ground-Up
    if project_type in ("Shell Building", "Ground-Up"):
        add_doc("geotech_report", "Geotechnical Report", True)
        add_doc("deferred_truss_form", "Deferred Truss Submittal Form", True)
        add_doc("special_inspections_form", "Special Inspections Form", True)
        add_doc("struct_calcs", "Structural Calculations", True)
        add_doc("struct_sheets", "Structural Sheets", True)

    # 3) Food-service-driven requirements
    analysis_food = bool(plan_analysis.get("is_food_service_by_plans", False))
    found = plan_analysis.get("found_items", {}) or {}
    any_food_feature = any(found.values())

    # Consider it food-service context if:
    # - user checked the box, OR
    # - plans clearly show food service, OR
    # - any of the specific food-service features are found
    is_food_context = (
        (project_type in ("Ground-Up", "Tenant Improvement (TI)"))
        and (is_food_service_checkbox or analysis_food or any_food_feature)
    )

    if is_food_context:
        note_base = "Required if plans show this food-service equipment or system."
        add_doc(
            "grease_interceptor",
            "Grease Interceptor Details/Submittal",
            required=bool(found.get("grease_interceptor", False)),
            note=note_base,
        )
        add_doc(
            "walk_in_cooler",
            "Walk-in Cooler Submittal/Details",
            required=bool(found.get("walk_in_cooler", False)),
            note=note_base,
        )
        add_doc(
            "walk_in_freezer",
            "Walk-in Freezer Submittal/Details",
            required=bool(found.get("walk_in_freezer", False)),
            note=note_base,
        )
        add_doc(
            "hood_specs",
            "Hood Specifications / Hood Schedule",
            required=bool(found.get("hood", False)),
            note=note_base,
        )
        add_doc(
            "iecc_docs",
            "IECC / Energy Code Compliance Documentation",
            required=bool(found.get("iecc_energy_code", False)),
            note="Required when IECC-specific details or energy compliance are called out.",
        )

    # Build summary / intake status
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

def main():
    st.set_page_config(
        page_title="City of Buckeye – Commercial Plan Intake (Beta)",
        layout="wide",
    )

    st.title("City of Buckeye – Commercial Plan Intake (Beta)")
    st.write(
        "Use this tool to perform an initial document intake check for COMMERCIAL projects "
        "(Shell, Ground-Up, or Tenant Improvement)."
    )

    # Sidebar: configuration / API key
    st.sidebar.header("Configuration")

    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = env_api_key
    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="If not set in environment, enter it here for this session.",
        )

    if not api_key:
        st.warning("Set OPENAI_API_KEY in the environment or enter an API key in the sidebar.")
        st.stop()

    client = get_client(api_key)
    st.sidebar.markdown(f"**Model:** `{MODEL_NAME}`")

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

    st.subheader("Documents Submitted with Application")

    col1, col2, col3 = st.columns(3)

    with col1:
        completed_application = st.checkbox("Completed Permit Application", value=True)
        arch_set = st.checkbox("Architectural Plan Set", value=True)
        mep_set = st.checkbox("MEP Plan Set", value=True)
        geotech_report = st.checkbox("Geotechnical Report")
        deferred_truss_form = st.checkbox("Deferred Truss Submittal Form")

    with col2:
        special_inspections_form = st.checkbox("Special Inspections Form")
        struct_calcs = st.checkbox("Structural Calculations")
        struct_sheets = st.checkbox("Structural Sheets")
        grease_interceptor = st.checkbox("Grease Interceptor Submittal/Details")
        walk_in_cooler = st.checkbox("Walk-in Cooler Submittal/Details")

    with col3:
        walk_in_freezer = st.checkbox("Walk-in Freezer Submittal/Details")
        hood_specs = st.checkbox("Hood Specifications / Hood Schedule")
        iecc_docs = st.checkbox("IECC / Energy Code Compliance Docs")

    docs_provided = {
        "completed_application": completed_application,
        "arch_set": arch_set,
        "mep_set": mep_set,
        "geotech_report": geotech_report,
        "deferred_truss_form": deferred_truss_form,
        "special_inspections_form": special_inspections_form,
        "struct_calcs": struct_calcs,
        "struct_sheets": struct_sheets,
        "grease_interceptor": grease_interceptor,
        "walk_in_cooler": walk_in_cooler,
        "walk_in_freezer": walk_in_freezer,
        "hood_specs": hood_specs,
        "iecc_docs": iecc_docs,
    }

    st.markdown("---")

    run_button = st.button("Run Intake Check", type="primary")

    if run_button:
        if not plan_pdf:
            st.error("Please upload the full plan set PDF before running the intake check.")
            st.stop()

        with st.spinner("Analyzing plans and building intake checklist..."):
            # Save uploaded PDF to temp, extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(plan_pdf.read())
                tmp_path = tmp.name

            try:
                plan_text = extract_pdf_text(tmp_path)
            except Exception as e:
                st.error(f"Error extracting text from PDF: {e}")
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                return
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            # Analyze food-service features via GPT
            try:
                plan_analysis = analyze_plans_for_food_service(
                    client, plan_text, project_type
                )
            except Exception as e:
                st.error(f"Error analyzing plans with AI: {e}")
                return

            # Build intake checklist
            checklist = build_intake_checklist(
                project_type=project_type,
                is_food_service_checkbox=is_food_service_checkbox,
                plan_analysis=plan_analysis,
                docs_provided=docs_provided,
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

        st.subheader("Plan Analysis (Food-Service Signals)")
        st.json(checklist["plan_analysis"])

        st.caption(
            "This tool performs an intake completeness check only. "
            "It does not constitute a full code compliance review."
        )


if __name__ == "__main__":
    main()
