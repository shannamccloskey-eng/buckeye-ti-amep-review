import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm


# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-4.1"  # or your preferred model

# GPT-5.1 pricing assumptions (example values – adjust to your actual pricing)
GPT51_INPUT_PRICE_PER_M = 15.00  # USD per 1M input tokens
GPT51_OUTPUT_PRICE_PER_M = 60.00  # USD per 1M output tokens


# ---------------- OPENAI CLIENT ----------------

def get_client(api_key: str) -> OpenAI:
    """
    Create an OpenAI client with the given API key.
    """
    return OpenAI(api_key=api_key)


# ---------------- PROMPT ----------------

BUCKEYE_TI_AMEP_PROMPT = """
You are a City of Buckeye Building Safety Plans Examiner performing an
Architectural, Mechanical, Electrical, Plumbing (AMEP) plan review for a
Tenant Improvement (TI) project within an existing building.

Review the uploaded TI plan set and provide:
1. A brief project scope summary.
2. A structured discrepancy table.
3. EnerGov-compatible comments.
4. A clear final determination (Approved, Approved with Comments, or Corrections Required).

Only use the following codes (by exact name and year) for references:
- 2024 International Building Code (IBC)
- 2024 International Mechanical Code (IMC)
- 2024 International Plumbing Code (IPC)
- 2024 International Fire Code (IFC)
- 2017 ICC A117.1
- 2023 National Electrical Code (NEC, NFPA 70)
- 2018 International Energy Conservation Code (IECC)
- ADA Standards for Accessible Design
- City of Buckeye Amendments

Classify each discrepancy by discipline (one of):
- Architectural
- Mechanical
- Electrical
- Plumbing
- Fire & Life Safety
- Accessibility
- Energy

The discrepancy table must be in GitHub-flavored Markdown format with exactly
the following columns in this order:

| Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction |

Each discrepancy row should:
- Reference at least one specific plan sheet (if possible).
- Include at least one specific code section (if possible).
- Clearly state a concise required correction in inspector language.

For EnerGov-compatible comments, use this format, one per line:
[Discipline] – [Code Reference]: [Required Correction].

For example:
Architectural – IBC 2024 §1005.3: Provide compliant egress width based on occupant load.

Finish with a Final Summary & Determination, including:
- "Determination: Approved", "Approved with Comments", or "Corrections Required".
- A brief closing statement:
  "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC,
   2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and City of Buckeye Amendments."

Tone: professional, concise, authoritative.
Avoid casual language or conversational fillers.
"""


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.

    1. Try direct text extraction with pypdf (for digital/searchable PDFs).
    2. If that fails (e.g., scanned/image-only PDFs), fall back to OCR using
       pdf2image + pytesseract, if installed.
    """
    # --- First pass: normal text extraction ---
    reader = PdfReader(pdf_path)
    all_text: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            all_text.append(page_text)

    text = "\n\n".join(all_text).strip()
    if text:
        return text

    # --- Fallback: OCR for scanned/image PDFs ---
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        # OCR libraries not available
        raise ValueError(
            "No extractable text found in PDF, and OCR dependencies "
            "(pdf2image, pytesseract) are not installed. "
            "Install these packages or upload a text/searchable PDF."
        )

    try:
        # Convert PDF pages to images (300 dpi is a good OCR baseline)
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        raise ValueError(
            f"No extractable text found in PDF, and image conversion for OCR failed: {e}"
        )

    ocr_chunks: List[str] = []
    for img in images:
        try:
            ocr_chunks.append(pytesseract.image_to_string(img))
        except Exception:
            # Skip any page that fails OCR, continue with others
            continue

    ocr_text = "\n\n".join(ocr_chunks).strip()
    if not ocr_text:
        raise ValueError(
            "No extractable text found in PDF, even after OCR. "
            "This appears to be a low-quality scan. Please upload a "
            "higher-quality or text-based plan set."
        )

    return ocr_text


# ---------------- TEXT HELPERS ----------------

def _split_paragraphs_from_lines(lines: List[str]) -> List[str]:
    """
    Turn a list of lines into paragraphs separated by blank lines.
    """
    paragraphs: List[str] = []
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


def _extract_markdown_table(
    all_lines: List[str],
    header_prefix: str = "| Sheet Reference",
) -> (List[str], List[str], List[str]):
    """
    Given a list of lines, attempt to locate the discrepancy table in Markdown.
    We look for a header starting with `header_prefix` and capture from there
    through the end of the contiguous Markdown table.
    """
    header_index: Optional[int] = None
    for i, line in enumerate(all_lines):
        if line.strip().startswith(header_prefix):
            header_index = i
            break

    if header_index is None:
        # No table found; everything is "pre" text
        return all_lines, [], []

    pre_lines = all_lines[:header_index]

    table_lines: List[str] = []
    for line in all_lines[header_index:]:
        stripped = line.strip()
        if not stripped:
            # blank line ends the table
            break
        if stripped.startswith("|"):
            table_lines.append(line)
        else:
            # first non-table, non-blank line after header ends the table
            break

    post_start = header_index + len(table_lines)
    post_lines = all_lines[post_start:]

    return pre_lines, table_lines, post_lines


def _markdown_table_to_data(table_lines: List[str]) -> List[List[str]]:
    """
    Convert a minimal GFM-style table into a list-of-lists of cell text.
    We ignore the alignment row like:
    | --- | --- | --- |
    Returns data rows only (no header), one list per row.
    """
    if not table_lines:
        return []

    lines = [ln for ln in table_lines if ln.strip()]
    if len(lines) < 2:
        return []

    # Skip header (lines[0]) and alignment (lines[1]).
    data_lines = lines[2:]

    table_data: List[List[str]] = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            # End of table
            break
        row_cells = [cell.strip() for cell in stripped.split("|")]
        # Remove empty cells from leading/trailing splits
        while row_cells and row_cells[0] == "":
            row_cells.pop(0)
        while row_cells and row_cells[-1] == "":
            row_cells.pop()
        if row_cells:
            table_data.append(row_cells)

    return table_data


# ---------------- PDF REPORT GENERATION ----------------

def save_text_as_pdf(text: str, pdf_path: Path) -> None:
    """
    Create a landscape-letter PDF that contains ONLY the discrepancy table
    (parsed from the Markdown table in the model output), laid out to fit the
    page width with word-wrapped text in each cell.

    If no table is found, falls back to writing the raw text as paragraphs.
    """

    stylesheet = getSampleStyleSheet()
    normal_style = stylesheet["Normal"]

    # Split the text into lines so we can look for the discrepancy table
    all_lines = text.splitlines()

    # Extract markdown discrepancy table lines
    _, table_lines, _ = _extract_markdown_table(all_lines)
    table_data_raw = _markdown_table_to_data(table_lines) if table_lines else []

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

    story: List[Any] = []

    if table_data_raw:
        # Fixed header row for the discrepancy table
        header_row = [
            "Sheet Reference",
            "Discipline",
            "Description of Issue",
            "Code Section",
            "Required Correction",
        ]

        # Use a smaller font and Paragraphs so word-wrap works cleanly
        cell_style = normal_style.clone("TableCell")
        cell_style.fontSize = 8
        cell_style.leading = 10

        def make_cell(content: str) -> Paragraph:
            # Clean up odd characters that may come from extraction (optional)
            content = (content or "").replace("■", "-")
            return Paragraph(content, cell_style)

        # Build table data with header + rows
        full_table_data: List[List[Any]] = []
        full_table_data.append([make_cell(c) for c in header_row])

        for row in table_data_raw:
            # Ensure each row has exactly 5 columns
            if len(row) < 5:
                row = row + [""] * (5 - len(row))
            elif len(row) > 5:
                row = row[:5]
            full_table_data.append([make_cell(c) for c in row])

        # Column widths sized to fit a landscape letter page (approx 235 mm total)
        col_widths = [
            30 * mm,  # Sheet Reference
            25 * mm,  # Discipline
            90 * mm,  # Description of Issue
            30 * mm,  # Code Section
            60 * mm,  # Required Correction
        ]

        table = Table(full_table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, "black"),
                    ("BACKGROUND", (0, 0), (-1, 0), "#DDDDDD"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )

        story.append(table)

    else:
        # Fallback: if no table is found, just drop the raw text in paragraphs
        for para in _split_paragraphs_from_lines(all_lines):
            story.append(Paragraph(para, normal_style))
            story.append(Spacer(1, 4 * mm))

    doc.build(story)


# ---------------- OPENAI CALL ----------------

def call_buckeye_ti_amep_single(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
) -> (str, Dict[str, Any]):
    """
    Send the full PDF text to the model in a single prompt, returning
    both the AI-written review and usage metadata.
    """
    system_message = BUCKEYE_TI_AMEP_PROMPT

    if project_description:
        user_content = (
            "PROJECT DESCRIPTION:\n"
            f"{project_description}\n\n"
            "FULL TI PLAN SET TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )
    else:
        user_content = (
            "FULL TI PLAN SET TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    review_text = response.choices[0].message.content or ""
    usage_data: Dict[str, Any] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    # Some newer models also expose reasoning tokens; if missing, default to 0
    reasoning_tokens = getattr(response.usage, "reasoning_tokens", 0) or 0
    usage_data["reasoning_tokens"] = reasoning_tokens
    return review_text, usage_data


# ---------------- HIGH-LEVEL REVIEW PIPELINE ----------------

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
        pdf_text = pdf_text[:800_000]
        pdf_text += "\n\n[NOTE TO REVIEWER: Plan text truncated to first 800,000 characters for token limits.]"

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
        "Upload Tenant Improvement (TI) plan set PDF(s) to generate an "
        "AMEP review (Architectural, Mechanical, Electrical, Plumbing, Fire, "
        "Accessibility, Energy)."
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
        st.warning(
            "Set OPENAI_API_KEY in the environment or enter an API key in the sidebar."
        )
        st.stop()

    client = get_client(api_key)

    st.sidebar.markdown(f"**Model:** `{MODEL_NAME}`")

    # Allow multiple PDF uploads
    uploaded_files = st.file_uploader(
        "Upload TI Plan Set PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Let user choose which PDF to review (important when multiple files)
    main_file = None
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        selected_name = st.selectbox(
            "Select which PDF to run the TI AMEP review on:",
            file_names,
            index=0,
        )
        for f in uploaded_files:
            if f.name == selected_name:
                main_file = f
                break

    project_description = st.text_area(
        "Optional project description",
        placeholder=(
            "Example: B occupancy office TI, no structural changes, "
            "new restrooms and lighting only."
        ),
    )

    run_button = st.button("Run TI AMEP Review", type="primary")

    if run_button:
        # Require at least one file and a selected main_file
        if not uploaded_files or main_file is None:
            st.error(
                "Please upload at least one PDF and select a primary plan set "
                "before running the review."
            )
            st.stop()

        # Progress bar + status text
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        progress_bar.progress(5)
        status_placeholder.text(
            f"Uploading and preparing file: {main_file.name} ..."
        )

        # Save selected file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(main_file.read())
            tmp_path = tmp.name

        try:
            progress_bar.progress(25)
            status_placeholder.text(
                "Step 1/3 – Extracting PDF text and sending for review ..."
            )

            review_text, usage_summary = run_review_pipeline_single(
                client,
                tmp_path,
                project_description.strip() or None,
            )

            progress_bar.progress(70)
            status_placeholder.text(
                "Step 2/3 – Formatting discrepancy table into PDF ..."
            )

            # Generate review PDF in temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix="_buckeye_ti_amep_review.pdf",
            ) as out_tmp:
                out_pdf_path = Path(out_tmp.name)

            save_text_as_pdf(review_text, out_pdf_path)

            with open(out_pdf_path, "rb") as f:
                pdf_bytes = f.read()

            try:
                os.remove(out_pdf_path)
            except OSError:
                pass

            progress_bar.progress(100)
            status_placeholder.text("Step 3/3 – TI AMEP review complete.")

        except Exception as e:
            st.error(f"Error during TI AMEP review: {e}")
            progress_bar.empty()
            status_placeholder.empty()
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

        # Optional: show raw review text in an expander
        with st.expander("Show full AI review text"):
            st.text_area(
                "Review output",
                value=review_text,
                height=400,
            )

        # Download button uses the primary file's name
        base_name = Path(main_file.name).stem
        download_name = f"{base_name}_buckeye_ti_amep_review.pdf"

        st.download_button(
            label="Download TI AMEP Review PDF",
            data=pdf_bytes,
            file_name=download_name,
            mime="application/pdf",
        )

        # Show token usage + cost
        if usage_summary:
            st.subheader("Token usage & cost estimate")
            st.json(usage_summary)

            if "cost_total_usd" in usage_summary:
                st.markdown(
                    f"**Estimated API cost (GPT-5.1): "
                    f"${usage_summary['cost_total_usd']:.4f} USD** "
                    f"(input: ${usage_summary['cost_input_usd']:.4f}, "
                    f"output: ${usage_summary['cost_output_usd']:.4f})"
                )

    # Footer caption (always shown)
    st.caption(
        "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC, "
        "2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and "
        "City of Buckeye Amendments."
    )


if __name__ == "__main__":
    main()
