import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import csv
import datetime

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors


# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-4.1"  # or your preferred model

GPT51_INPUT_PRICE_PER_M = 15.00  # USD per 1M input tokens
GPT51_OUTPUT_PRICE_PER_M = 60.00  # USD per 1M output tokens


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


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

Finish with a Final Summary & Determination, including a clear determination
and closing statement.
"""


# ---------------- PDF / TEXT HELPERS ----------------
def save_feedback_csv(
    csv_path: Path,
    tool_name: str,
    run_id: str,
    filename: str,
    rating: str,
    comments: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a single feedback record to a CSV file.

    - csv_path: Path to CSV file (e.g., feedback_ti_amep.csv)
    - tool_name: Short name of the tool ("TI_AMEP", "GEO_SUMMARY", etc.)
    - run_id: Unique id per run (e.g., ISO timestamp or uuid)
    - filename: Source PDF file name
    - rating: e.g., 'Looks good', 'Mostly okay', 'Needs corrections'
    - comments: Free-form reviewer comments
    - extra: Optional dict for extra metadata (tokens, model, etc.)
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()

    fieldnames = [
        "timestamp_utc",
        "tool_name",
        "run_id",
        "filename",
        "rating",
        "comments",
    ]

    extra = extra or {}
    for k in extra.keys():
        if k not in fieldnames:
            fieldnames.append(k)

    timestamp = datetime.datetime.utcnow().isoformat()

    row: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "tool_name": tool_name,
        "run_id": run_id,
        "filename": filename,
        "rating": rating,
        "comments": comments,
    }
    row.update(extra)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            all_text.append(page_text)
    return "\n\n".join(all_text).strip()


def _split_paragraphs_from_lines(lines: List[str]) -> List[str]:
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
) -> Tuple[List[str], List[str], List[str]]:
    """
    Locate the Markdown discrepancy table inside the AI output and
    return (pre_lines, table_lines, post_lines).
    """
    header_index: Optional[int] = None
    for i, line in enumerate(all_lines):
        if line.strip().startswith(header_prefix):
            header_index = i
            break

    if header_index is None:
        return all_lines, [], []

    pre_lines = all_lines[:header_index]

    table_lines: List[str] = []
    for line in all_lines[header_index:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("|"):
            table_lines.append(line)
        else:
            break

    post_start = header_index + len(table_lines)
    post_lines = all_lines[post_start:]
    return pre_lines, table_lines, post_lines


def _markdown_table_to_data(table_lines: List[str]) -> List[List[str]]:
    """
    Convert the GitHub-flavored discrepancy table into a 2D list of strings.
    """
    if not table_lines:
        return []

    lines = [ln for ln in table_lines if ln.strip()]
    if len(lines) < 2:
        return []

    # Skip header & separator; data starts after that
    data_lines = lines[2:]

    table_data: List[List[str]] = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        row_cells = [cell.strip() for cell in stripped.split("|")]
        while row_cells and row_cells[0] == "":
            row_cells.pop(0)
        while row_cells and row_cells[-1] == "":
            row_cells.pop()
        if row_cells:
            table_data.append(row_cells)

    return table_data


def save_text_as_pdf(
    text: str,
    pdf_path: Path,
    original_filename: str,
    logo_name: str = "City of Buckeye 2025.png",
) -> None:
    """
    Create a Buckeye-branded landscape PDF with:
    - Logo (if present)
    - Title: City of Buckeye – TI AMEP Review
    - Subtitle with original TI plan-set filename
    - Discrepancy table as a wrapped landscape table

    If no table is detected, falls back to paragraphs.
    """
    stylesheet = getSampleStyleSheet()
    normal_style = stylesheet["Normal"]

    all_lines = text.splitlines()
    _, table_lines, _ = _extract_markdown_table(all_lines)
    table_data_raw = _markdown_table_to_data(table_lines) if table_lines else []

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

    # ---- Header: logo + title + subtitle ----
    logo_path = Path(__file__).parent / logo_name
    if logo_path.exists():
        img = RLImage(str(logo_path))
        img.drawHeight = 50
        img.drawWidth = 150
        story.append(img)
        story.append(Spacer(1, 8 * mm))

    title = "City of Buckeye – TI AMEP Review"
    story.append(Paragraph(title, stylesheet["Title"]))
    story.append(Spacer(1, 3 * mm))

    subtitle = f"Review generated from TI plan set: {original_filename}"
    story.append(Paragraph(subtitle, normal_style))
    story.append(Spacer(1, 8 * mm))

    # ---- Discrepancy table (preferred) ----
    if table_data_raw:
        header_row = [
            "Sheet Reference",
            "Discipline",
            "Description of Issue",
            "Code Section",
            "Required Correction",
        ]

        # Smaller font + wrapping paragraph style
        cell_style = ParagraphStyle(
            "TableCell",
            parent=normal_style,
            fontSize=8,
            leading=10,
        )

        def make_cell(content: str) -> Paragraph:
            content = (content or "").replace("■", "-").replace("\n", " ")
            return Paragraph(content, cell_style)

        full_table_data: List[List[Any]] = []
        full_table_data.append([make_cell(c) for c in header_row])

        for row in table_data_raw:
            if len(row) < 5:
                row = row + [""] * (5 - len(row))
            elif len(row) > 5:
                row = row[:5]
            full_table_data.append([make_cell(c) for c in row])

        # Column widths proportional to page width
        total_width = pagesize[0] - doc.leftMargin - doc.rightMargin
        col_widths = [
            total_width * 0.11,  # Sheet Ref
            total_width * 0.12,  # Discipline
            total_width * 0.43,  # Description
            total_width * 0.12,  # Code Section
            total_width * 0.22,  # Required Correction
        ]

        table = Table(full_table_data, colWidths=col_widths, repeatRows=1)

        header_color = colors.HexColor("#c45c26")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), header_color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                    ("TOPPADDING", (0, 0), (-1, 0), 4),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("TOPPADDING", (0, 1), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )

        # Light row striping for readability
        for r_idx in range(1, len(full_table_data)):
            if r_idx % 2 == 0:
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, r_idx), (-1, r_idx), colors.whitesmoke),
                        ]
                    )
                )

        story.append(table)

    else:
        # Fallback: just dump paragraphs (still with header + logo)
        for para in _split_paragraphs_from_lines(all_lines):
            story.append(Paragraph(para, normal_style))
            story.append(Spacer(1, 4 * mm))

    doc.build(story)


# ---------------- OPENAI PIPELINE ----------------

def call_buckeye_ti_amep_single(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
) -> (str, Dict[str, Any]):
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

    reasoning_tokens = getattr(response.usage, "reasoning_tokens", 0) or 0
    usage_data["reasoning_tokens"] = reasoning_tokens
    return review_text, usage_data


def run_review_pipeline_single(
    client: OpenAI,
    pdf_path: str,
    project_description: Optional[str],
) -> (str, Dict[str, Any]):
    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text.strip():
        raise ValueError("No extractable text found in PDF.")

    MAX_CHARS = 80_000
    if len(pdf_text) > MAX_CHARS:
        pdf_text = pdf_text[:MAX_CHARS]
        pdf_text += (
            "\n\n[NOTE TO REVIEWER: Plan text truncated to the first "
            f"{MAX_CHARS} characters to stay within model rate limits.]"
        )

    review_text, usage = call_buckeye_ti_amep_single(
        client,
        full_pdf_text=pdf_text,
        project_description=project_description,
    )

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0) or 0)
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0) or 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    total_tokens = usage.get(
        "total_tokens",
        input_tokens + output_tokens + reasoning_tokens,
    )

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

def main(embed: bool = False):
    """
    TI AMEP Review UI.
    embed = False → standalone; embed = True → called from master app.
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – TI AMEP Review (Beta)",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – TI AMEP Review (Beta)")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.write(
        "Upload Tenant Improvement (TI) plan set PDF(s) to generate an "
        "AMEP review (Architectural, Mechanical, Electrical, Plumbing, Fire, "
        "Accessibility, Energy)."
    )

    # API key from environment only
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the app."
        )
        st.stop()

    client = get_client(env_api_key)

    uploaded_files = st.file_uploader(
        "Upload TI Plan Set PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

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
        if not uploaded_files or main_file is None:
            st.error(
                "Please upload at least one PDF and select a primary plan set "
                "before running the review."
            )
            st.stop()

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        progress_bar.progress(5)
        status_placeholder.text(
            f"Uploading and preparing file: {main_file.name} ..."
        )

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

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix="_buckeye_ti_amep_review.pdf",
            ) as out_tmp:
                out_pdf_path = Path(out_tmp.name)

            save_text_as_pdf(
                review_text,
                out_pdf_path,
                original_filename=main_file.name,
            )

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

        with st.expander("Show full AI review text"):
            st.text_area(
                "Review output",
                value=review_text,
                height=400,
            )

        base_name = Path(main_file.name).stem
        download_name = f"{base_name}_buckeye_ti_amep_review.pdf"

        st.download_button(
            label="Download TI AMEP Review PDF",
            data=pdf_bytes,
            file_name=download_name,
            mime="application/pdf",
        )

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


if __name__ == "__main__":
    main()
