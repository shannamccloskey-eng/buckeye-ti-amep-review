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
- 2010 ADA Standards for Accessible Design (ADA)
- City of Buckeye Amendments

Do NOT quote code text verbatim.
You may cite code sections as, for example, "IBC 2024 §1005.3".

Organize your review into these disciplines:
- Architectural
- Mechanical
- Electrical
- Plumbing
- Fire & Life Safety
- Accessibility
- Energy

The discrepancy table must be in GitHub-flavored Markdown format with **exactly**
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


# ---------------- PDF UTILITIES ----------------

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


def _extract_markdown_table(
    all_lines: List[str],
    header_prefix: str = "| Sheet Reference",
) -> (List[str], List[str], List[str]):
    """
    Given a list of lines, attempt to locate the discrepancy table in Markdown.
    We look for a header starting with `header_prefix` and capture from there
    through the end of a Markdown table.
    """
    header_index = None
    for i, line in enumerate(all_lines):
        if line.strip().startswith(header_prefix):
            header_index = i
            break

    if header_index is None:
        return all_lines, [], []

    table_lines: List[str] = []
    table_started = False

    for line in all_lines[header_index:]:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped:
            table_lines.append(line)
            table_started = True
        else:
            if table_started:
                # End of table
                break

    pre_lines = all_lines[:header_index]
    post_start = header_index + len(table_lines)
    post_lines = all_lines[post_start:]

    return pre_lines, table_lines, post_lines


def _markdown_table_to_data(table_lines: List[str]) -> List[List[str]]:
    """
    Convert a minimal GFM-style table into a list-of-lists of cell text.
    We ignore the alignment row like:
    | --- | --- | --- |
    """
    if not table_lines:
        return []

    lines = [ln for ln in table_lines if ln.strip()]
    if len(lines) < 2:
        return []

    header = lines[0]
    data_lines = lines[1:]

    table_data: List[List[str]] = []
    for i, line in enumerate(data_lines):
        if set(line.strip()) <= {"|", "-", " "}:
            # alignment row, skip
            continue
        row_cells = [cell.strip() for cell in line.split("|")]
        # Remove empty cells from leading/trailing splits
        while row_cells and row_cells[0] == "":
            row_cells.pop(0)
        while row_cells and row_cells[-1] == "":
            row_cells.pop()
        if row_cells:
            table_data.append(row_cells)

    return table_data


def save_text_as_pdf(text: str, pdf_path: Path) -> None:
    """
    Create a landscape-letter PDF containing the review text, plus any
    Markdown discrepancy table rendered as a simple table.
    """
    # Prepare styles
    stylesheet = getSampleStyleSheet()
    normal_style = stylesheet["Normal"]
    heading_style = stylesheet["Heading2"]

    # Split the text into lines so we can look for the discrepancy table
    all_lines = text.splitlines()

    # Try to extract markdown discrepancy table
    pre_lines, table_lines, post_lines = _extract_markdown_table(all_lines)
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

    # 1) Everything before the table as paragraphs
    pre_text = "\n".join(pre_lines).strip()
    if pre_text:
        for para in _split_paragraphs_from_lines(pre_text.splitlines()):
            story.append(Paragraph(para, normal_style))
            story.append(Spacer(1, 4 * mm))

    # 2) Discrepancy table (if any)
    if table_data_raw:
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Discrepancy Table", heading_style))
        story.append(Spacer(1, 3 * mm))

        # Optionally clamp overly long cells for PDF readability
        max_cell_length = 350

        processed_rows: List[List[str]] = []
        for row in table_data_raw:
            new_row: List[str] = []
            for cell in row:
                if len(cell) > max_cell_length:
                    cell = cell[: max_cell_length - 3] + "..."
                new_row.append(cell)
            processed_rows.append(new_row)

        table = Table(processed_rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, "black"),
                    ("BACKGROUND", (0, 0), (-1, 0), "#DDDDDD"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 6 * mm))

    # 3) Everything after the table as normal paragraphs
    post_text = "\n".join(post_lines).strip()
    if post_text:
        for para in _split_paragraphs_from_lines(post_text.splitlines()):
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
    usage_data = {
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
        status_placeholder.text(f"Starting TI AMEP review on: {main_file.name} ...")

        # Save selected file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(main_file.read())
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
            status_placeholder.text("TI AMEP review complete.")

        except Exception as e:
            st.error(
                "Error during review: "
                f"{e}\n\n"
                "If this is a scanned or image-only PDF, please export a "
                "searchable/text-based PDF from CAD or request a digital "
                "plan set from the applicant."
            )
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
