import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import textwrap

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-5.1"  # OpenAI API model ID

# Pricing for GPT-5.1 text tokens (per 1M tokens) – from OpenAI docs
GPT51_INPUT_PRICE_PER_M = 1.25   # USD per 1M input tokens
GPT51_OUTPUT_PRICE_PER_M = 10.0  # USD per 1M output tokens

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
- You MUST format the Discrepancy Table as a markdown table with this exact header row:
  | Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction |
- If the sheet reference is unknown from the excerpt, indicate "N/A" or "Not Shown".

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

def _split_paragraphs_from_lines(lines: List[str]) -> List[str]:
    """
    Turn a list of lines into paragraphs separated by blank lines.
    """
    paragraphs = []
    buffer: List[str] = []
    for line in lines:
        if line.strip() == "":
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
        else:
            buffer.append(line.strip())
    if buffer:
        paragraphs.append(" ".join(buffer).strip())
    return paragraphs

def _extract_markdown_table(lines: List[str]) -> (List[str], List[str], List[str]):
    """
    Find a markdown table that starts with the header row containing:
    '| Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction |'
    and return (pre_lines, table_lines, post_lines).
    If no table found, table_lines will be empty and all text is in pre_lines+post_lines.
    """
    header_substring = "Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction"
    start_idx = None

    for i, line in enumerate(lines):
        if header_substring in line:
            start_idx = i
            break

    if start_idx is None:
        # No table found
        return lines, [], []

    table_lines = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if "|" in line and line.strip():
            table_lines.append(line)
            i += 1
        else:
            break

    pre_lines = lines[:start_idx]
    post_lines = lines[i:]

    return pre_lines, table_lines, post_lines

def _markdown_table_to_data(table_lines: List[str]) -> List[List[str]]:
    """
    Convert markdown table lines into a list-of-lists for ReportLab Table.
    Skips separator rows like '|---|---|...|'.
    """
    data: List[List[str]] = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip separator line
        # e.g.: | --- | --- | ... |
        no_bars = stripped.replace("|", "").replace(":", "").replace("-", "").strip()
        if no_bars == "":
            continue

        # Split into cells
        parts = stripped.strip("|").split("|")
        row = [cell.strip() for cell in parts]
        data.append(row)
    return data

def save_text_as_pdf(text: str, pdf_path: Path):
    """
    Save review text as a formatted PDF:
    - Landscape Letter
    - Body text as paragraphs
    - Discrepancy Table rendered as a real table if markdown table is found
    """
    # Prepare lines
    all_lines = text.splitlines()

    # Try to extract markdown discrepancy table
    pre_lines, table_lines, post_lines = _extract_markdown_table(all_lines)
    table_data = _markdown_table_to_data(table_lines) if table_lines else []

    # Landscape letter document
    pagesize = landscape(letter)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=pagesize,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    body_style = styles["Normal"]
    body_style.fontName = "Helvetica"
    body_style.fontSize = 9
    body_style.leading = 11

    heading_style = styles["Heading3"]
    heading_style.fontName = "Helvetica-Bold"
    heading_style.fontSize = 11
    heading_style.leading = 14

    story: List[Any] = []

    # Preamble paragraphs
    pre_paragraphs = _split_paragraphs_from_lines(pre_lines)
    for para_text in pre_paragraphs:
        story.append(Paragraph(para_text, body_style))
        story.append(Spacer(1, 3 * mm))

    # Discrepancy Table, if present
    if table_data:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Discrepancy Table", heading_style))
        story.append(Spacer(1, 2 * mm))

        # If header row has 5 cols, we assume the expected structure
        col_widths = None
        if len(table_data[0]) == 5:
            # Compute column widths proportionally
            page_width = pagesize[0]
            effective_width = page_width - doc.leftMargin - doc.rightMargin
            col_widths = [
                0.12 * effective_width,  # Sheet Reference
                0.12 * effective_width,  # Discipline
                0.36 * effective_width,  # Description
                0.12 * effective_width,  # Code Section
                0.28 * effective_width,  # Required Correction
            ]

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl_style = TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, 0),  # thin grid, default color
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
            ]
        )
        tbl.setStyle(tbl_style)
        story.append(tbl)
        story.append(Spacer(1, 5 * mm))

    # Post-table paragraphs
    post_paragraphs = _split_paragraphs_from_lines(post_lines)
    if post_paragraphs:
        story.append(Spacer(1, 4 * mm))
        for para_text in post_paragraphs:
            story.append(Paragraph(para_text, body_style))
            story.append(Spacer(1, 3 * mm))

    # Build the document
    doc.build(story)

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
        "- You MUST format the Discrepancy Table as a markdown table with the header:\n"
        "  | Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction |\n"
        "- If specific sheet numbers are not visible, use 'N/A' or 'Not Shown'.\n\n"
        "FULL PDF TEXT:\n"
        f"{full_pdf_text}"
    )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=CITY_OF_BUCKEYE_INSTRUCTIONS,
        input=user_prompt,
    )
    usage_dict = _extract_usage_dict(response)
    return response.output_text, usage_dict

def run_review_pipeline_single(
    client: OpenAI,
    pdf_path: str,
    project_description: Optional[str],
) -> (str, Dict[str, Any]):
    """
    Extract text from the PDF and run a single OpenAI call (no chunking),
    returning both the review text and a usage + cost summary.
    """

    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text.strip():
        raise ValueError("No extractable text found in PDF.")

    # Safety cap to avoid absurdly large prompts
    if len(pdf_text) > 800_000:
        raise ValueError(
            "PDF text is very large. For now, please test with a smaller TI PDF "
            "or split the document. (Chunking has been disabled in this version.)"
        )

    review_text, usage = call_buckeye_ti_amep_single(
        client,
        full_pdf_text=pdf_text,
        project_description=project_description,
    )

    # Token counts
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0) or 0)
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0) or 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    total_tokens = usage.get(
        "total_tokens",
        input_tokens + output_tokens + reasoning_tokens,
    )

    # Cost estimate (USD) based on GPT-5.1 pricing
    cost_input = (input_tokens / 1_000_000.0) * GPT51_INPUT_PRICE_PER_M
    cost_output = (output_tokens / 1_000_000.0) * GPT51_OUTPUT_PRICE_PER_M
    cost_total = cost_input + cost_output

    usage_summary: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_input_usd": round(cost_input, 6),
        "cost_output_usd": round(cost_output, 6),
        "cost_total_usd": round(cost_total, 6),
        "pricing_model": "GPT-5.1 standard text tokens",
        "pricing_notes": (
            "Cost estimate uses official GPT-5.1 prices: "
            f"${GPT51_INPUT_PRICE_PER_M}/1M input, "
            f"${GPT51_OUTPUT_PRICE_PER_M}/1M output tokens."
        ),
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

        # Progress bar + status text
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        progress_bar.progress(5)
        status_placeholder.text("Starting TI AMEP review...")

        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            progress_bar.progress(25)
            status_placeholder.text("Extracting PDF text and preparing review request...")

            review_text, usage_summary = run_review_pipeline_single(
                client,
                tmp_path,
                project_description.strip() or None,
            )

            progress_bar.progress(70)
            status_placeholder.text("Generating formatted review PDF...")

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

            progress_bar.progress(100)
            status_placeholder.text("TI AMEP review complete.")

        except Exception as e:
            st.error(f"Error during review: {e}")
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

        # Show token usage + cost
        st.subheader("Token usage & cost estimate")
        st.json(usage_summary)

        if "cost_total_usd" in usage_summary:
            st.markdown(
                f"**Estimated API cost (GPT-5.1): "
                f"${usage_summary['cost_total_usd']:.4f} USD** "
                f"(input: ${usage_summary['cost_input_usd']:.4f}, "
                f"output: ${usage_summary['cost_output_usd']:.4f})"
            )

        st.caption(
            "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC, "
            "2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and City of Buckeye Amendments."
        )


if __name__ == "__main__":
    main()
