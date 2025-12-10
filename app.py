import os
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import textwrap

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-5.1"  # OpenAI API model ID

CITY_OF_BUCKEYE_INSTRUCTIONS = """
You are "City of Buckeye CB3PO". You represent the City of Buckeye Building
Safety / Building Department and serve as a Building Safety Plans Examiner.

SCOPE:
You are performing Architectural, Mechanical, Electrical, Plumbing (AMEP) plan
reviews specifically for Tenant Improvement (TI) projects only.

ADOPTED CODES (reference only; do NOT quote text):
- 2024 International Building Code (IBC)
- 2024 International Mechanical Code (IMC)
- 2024 International Plumbing Code (IPC)
- 2024 International Fire Code (IFC)
- 2017 ICC A117.1
- 2018 International Energy Conservation Code (IECC)
- 2023 National Electrical Code (NEC / NFPA 70)
- ADA Standards for Accessible Design
- City of Buckeye Amendments

TI AMEP REVIEW WORKFLOW (Run TI AMEP Review):

STEP 1 — Identify Project Scope
- Confirm the project is a Tenant Improvement (TI).
- Classify the review into: Architectural, Mechanical, Electrical, Plumbing,
  Fire, Accessibility, and Energy.

STEP 2 — Conduct Full AMEP Review

A. Architectural (IBC 2024, IFC 2024, IECC 2018)
- Confirm occupancy classification, construction type, fire-resistance and
  required separations.
- Verify means of egress, occupant load, exit signs, emergency lighting,
  door hardware, corridor ratings (if required).
- Review restrooms, accessibility, finishes, fire protection coordination,
  and energy-related envelope items as applicable to a TI.

B. Mechanical (IMC 2024 + IECC 2018)
- Check ventilation and outside air, exhaust, duct construction and routing,
  fire/smoke dampers where required, and equipment access/clearances.
- Verify controls and any energy code-related mechanical provisions.

C. Plumbing (IPC 2024 + ADA + ICC A117.1-2017)
- Verify plumbing fixture counts for occupancy, fixture locations, accessible
  fixtures, clearances, grab bars, lavatories, and water closets.
- Check water supply, drainage, venting, backflow prevention, indirect waste,
  grease interceptors (if applicable), and water heater safety.

D. Electrical (NEC 2023 + IECC + ADA/A117.1)
- Confirm branch circuits, overcurrent protection, panel schedules,
  grounding/bonding, receptacle spacing where relevant, and coordination with
  mechanical and plumbing equipment.
- Verify egress lighting, exit signage power, lighting controls, and any
  energy code requirements for lighting.

E. Fire & Life Safety (IFC 2024 + IBC Ch. 9)
- Coordinate sprinkler and fire alarm design (if provided) with TI layout.
- Check egress lighting, exit signs, fire-extinguisher locations, and
  hazardous materials (if any).

STEP 3 — Discrepancy Table
- Produce a table with columns:
  Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction.
- If the sheet reference is unknown from the excerpt, indicate "N/A" or
  "Not Shown".

STEP 4 — EnerGov-Compatible Comments
- Format each comment as:
  [Discipline] – [Code Reference]: [Required Correction].
  Example: Architectural – IBC 2024 §1005.3: Provide compliant egress width
  based on occupant load.

STEP 5 — Final Summary & Determination
- Provide one of:
  - Approved
  - Approved with Comments
  - Corrections Required.
- Include the closing statement:
  "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC,
   2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and City of
   Buckeye Amendments. All cited code sections are publicly available
   on ICCsafe.org, ADA.gov, and NFPA.org."

GENERAL BEHAVIOR:
- Tone: professional, concise, and authoritative; municipal plan review style.
- Do NOT quote or reproduce code text; cite section numbers only.
- Focus ONLY on TI AMEP scope; do not perform structural or geotechnical review.
"""

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

def save_text_as_pdf(text: str, pdf_path: Path):
    """
    Save a long plain-text report as a simple, formatted PDF.
    """
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    left_margin = 72
    top_margin = 72
    bottom_margin = 72
    max_width_chars = 100
    line_height = 12

    lines = []
    for raw_line in text.splitlines():
        if raw_line.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(raw_line, max_width_chars) or [""]
            lines.extend(wrapped)

    y = height - top_margin
    c.setFont("Helvetica", 10)

    for line in lines:
        if y < bottom_margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - top_margin
        c.drawString(left_margin, y, line)
        y -= line_height

    c.save()

# ---------------- OPENAI HELPERS ----------------

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def _extract_usage_dict(response) -> Dict[str, Any]:
    usage_dict: Dict[str, Any] = {}
    usage = getattr(response, "usage", None)
    if usage is None:
        return usage_dict

    keys = [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ]
    for key in keys:
        val = getattr(usage, key, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(key)
        if val is not None:
            usage_dict[key] = val
    return usage_dict

def call_buckeye_ti_amep_single(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
) -> (str, Dict[str, Any]):
    """
    Single OpenAI call for the entire extracted PDF text (no chunking).
    """
    project_line = ""
    if project_description:
        project_line = f"PROJECT DESCRIPTION (TI): {project_description}\n\n"

    user_prompt = (
        "Run TI AMEP Review for this Tenant Improvement (TI) permit submittal.\n"
        "You are performing Architectural, Mechanical, Electrical, Plumbing, Fire, "
        "Accessibility, and Energy review only, per your TI AMEP workflow.\n\n"
        f"{project_line}"
        "REVIEW REQUIREMENTS:\n"
        "- Treat this as the full TI plan set text extracted from the PDF.\n"
        "- Identify discrepancies, missing information, and code issues.\n"
        "- Produce a discrepancy table and EnerGov-compatible comments.\n"
        "- If specific sheet numbers are not visible, use 'N/A' or 'Not Shown'.\n\n"
        "FULL PDF TEXT:\n"
        f"{full_pdf_text}"
    )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=CITY_OF_BUCKEYE_INSTRUCTIONS,
        input=user_prompt,
        # Optional: you can bound max_output_tokens if you want
        # max_output_tokens=4000,
    )
    usage_dict = _extract_usage_dict(response)
    return response.output_text, usage_dict

def run_review_pipeline_single(
    client: OpenAI,
    pdf_path: str,
    project_description: Optional[str],
) -> (str, Dict[str, int]):
    """
    Extract text from the PDF and run a single OpenAI call (no chunking).
    """

    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text.strip():
        raise ValueError("No extractable text found in PDF.")

    # Simple safety cap: avoid sending an absurdly large prompt that could hang or be rejected.
    # Roughly, 800,000 characters ~ 200k tokens for English text (very rough).
    if len(pdf_text) > 800_000:
        raise ValueError(
            "PDF text is very large. For now, please test with a smaller TI PDF "
            "or split the document. (Chunking has been disabled by request.)"
        )

    review_text, usage = call_buckeye_ti_amep_single(
        client,
        full_pdf_text=pdf_text,
        project_description=project_description,
    )

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    total_tokens = usage.get(
        "total_tokens",
        input_tokens + output_tokens + reasoning_tokens,
    )

    usage_summary = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }

    return review_text, usage_summary

# ---------------- STREAMLIT UI ----------------

def main():
    st.set_page_config(
        page_title="City of Buckeye – TI AMEP Review (Beta)",
        layout="wide",
    )

    st.title("City of Buckeye – TI AMEP Review (Beta)")
    st.write(
        "Upload a Tenant Improvement (TI) plan set in PDF format to generate an "
        "AMEP review (Architectural, Mechanical, Electrical, Plumbing, Fire, Accessibility, Energy)."
    )

    # Sidebar: API key and model info
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

    uploaded_file = st.file_uploader("Upload TI Plan Set (PDF)", type=["pdf"])

    project_description = st.text_area(
        "Optional project description",
        placeholder="Example: B occupancy office TI, no structural changes, new restrooms and lighting only.",
    )

    run_button = st.button("Run TI AMEP Review", type="primary")

    if run_button:
        if not uploaded_file:
            st.error("Please upload a PDF file before running the review.")
            st.stop()

        with st.spinner("Running TI AMEP Review (single-pass, no chunking)..."):
            # Save uploaded file to a temp path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                review_text, usage_summary = run_review_pipeline_single(
                    client,
                    tmp_path,
                    project_description.strip() or None,
                )
            except Exception as e:
                st.error(f"Error during review: {e}")
                return
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            # Generate review PDF in temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix="_buckeye_ti_amep_review.pdf") as out_tmp:
                out_pdf_path = Path(out_tmp.name)

            save_text_as_pdf(review_text, out_pdf_path)

            with open(out_pdf_path, "rb") as f:
                pdf_bytes = f.read()

            try:
                os.remove(out_pdf_path)
            except OSError:
                pass

        st.success("TI AMEP review complete.")

        # Download button
        base_name = Path(uploaded_file.name).stem
        download_name = f"{base_name}_buckeye_ti_amep_review.pdf"

        st.download_button(
            label="Download Review PDF",
            data=pdf_bytes,
            file_name=download_name,
            mime="application/pdf",
        )

        # Show token usage
        st.subheader("Token usage summary")
        st.json(usage_summary)

        st.caption(
            "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC, "
            "2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and City of Buckeye Amendments."
        )


if __name__ == "__main__":
    main()
