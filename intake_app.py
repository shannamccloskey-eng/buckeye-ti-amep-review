import os
import csv
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tempfile

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

MODEL_NAME = "gpt-4.1"


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


# ---------------- FEEDBACK HELPER ----------------

def save_feedback_csv(
    csv_path: Path,
    tool_name: str,
    run_id: str,
    filename: str,
    rating: str,
    comments: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
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


# ---------------- PROMPT ----------------

BUCKEYE_INTAKE_PROMPT = """
You are a City of Buckeye Building Safety Plans Examiner performing an
INTAKE COMPLETENESS REVIEW for a COMMERCIAL building plan submittal.

This is an INTAKE review only:
- Focus on whether required documents and information are present.
- Do NOT perform deep structural or detailed code checks.
- Assume 2024 IBC, 2018 IECC, and City of Buckeye amendments are in effect,
  but you should NOT quote code text.

You are given:
- The extracted text of the submitted plan set.
- An optional project description entered by the reviewer.

Your tasks:

1) Project & Scope Summary
   • Briefly describe the project (occupancy, stories, construction type if known).
   • Identify whether it appears to involve food service or hazardous uses.
   • Summarize any special features (e.g., new restrooms, mezzanines, commercial kitchen).

2) Intake Completeness Table
   Create a GitHub-flavored Markdown table with EXACTLY these columns:

   | Item | Required? | Provided? | Comments / Required Action |

   Include rows for the typical commercial submittal items, such as:
   - Application / cover sheet information (address, APN, contacts).
   - Code analysis (occupancy, construction type, allowable area, fire protection).
   - Site plan / civil plan.
   - Architectural floor plans & RCP.
   - Building elevations & sections.
   - Structural plans (foundation, framing, details).
   - Structural calculations.
   - Truss / joist / delegated design submittals (if applicable).
   - Geotechnical / soil report (if applicable).
   - Mechanical plans / schedules.
   - Plumbing plans / isometrics.
   - Electrical power / lighting plans.
   - Energy compliance (COMcheck or equivalent).
   - Fire protection (sprinkler/standpipe) shop drawings (if applicable).
   - Fire alarm drawings (if applicable).
   - Accessibility details (restrooms, parking, routes).
   - Special inspections statement / schedule (if applicable).
   - Other supporting documents clearly referenced in the set.

   For each row:
   - "Required?": Yes / No / Maybe (if dependent on scope).
   - "Provided?": Yes / No / Unclear from plans.
   - "Comments / Required Action": Short, directive language such as
     "Provide structural calculations for new steel canopy" or
     "Fire alarm shop drawings to be deferred submittal; note on cover sheet."

3) Intake Determination
   At the end of the report (after the table), provide a short narrative
   "Intake Determination" stating whether the submittal appears:
   - "Intake complete – OK to route for full review", or
   - "Intake incomplete – additional items required before routing", with a
     brief list of the highest-priority missing items.

Formatting:
- The table MUST appear in the response exactly once, with the header row
  and separator row, in GitHub-flavored Markdown.
- You may include short narrative text before and after the table, but the
  table must be easy to locate.
"""


# ---------------- PDF / TABLE HELPERS ----------------

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
    header_prefix: str = "| Item | Required?",
) -> Tuple[List[str], List[str], List[str]]:
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
    if not table_lines:
        return []

    lines = [ln for ln in table_lines if ln.strip()]
    if len(lines) < 2:
        return []

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

    logo_path = Path(__file__).parent / logo_name
    if logo_path.exists():
        img = RLImage(str(logo_path))
        img.drawHeight = 50
        img.drawWidth = 150
        story.append(img)
        story.append(Spacer(1, 8 * mm))

    title = "City of Buckeye – Commercial Plan Intake (Build 1.0)"
    story.append(Paragraph(title, stylesheet["Title"]))
    story.append(Spacer(1, 3 * mm))

    subtitle = f"Intake review generated from plan set: {original_filename}"
    story.append(Paragraph(subtitle, normal_style))
    story.append(Spacer(1, 8 * mm))

    if table_data_raw:
        header_row = [
            "Item",
            "Required?",
            "Provided?",
            "Comments / Required Action",
        ]

        cell_style = ParagraphStyle(
            "TableCell",
            parent=normal_style,
            fontSize=8,
            leading=10,
        )

        def make_cell(content: str) -> Paragraph:
            content = (content or "").replace("\n", " ")
            return Paragraph(content, cell_style)

        full_table_data: List[List[Any]] = []
        full_table_data.append([make_cell(c) for c in header_row])

        for row in table_data_raw:
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            elif len(row) > 4:
                row = row[:4]
            full_table_data.append([make_cell(c) for c in row])

        total_width = pagesize[0] - doc.leftMargin - doc.rightMargin
        col_widths = [
            total_width * 0.22,
            total_width * 0.12,
            total_width * 0.12,
            total_width * 0.54,
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

        for r_idx in range(1, len(full_table_data)):
            if r_idx % 2 == 0:
                table.setStyle(
                    TableStyle(
                        [("BACKGROUND", (0, r_idx), (-1, r_idx), colors.whitesmoke)]
                    )
                )

        story.append(table)
    else:
        for para in _split_paragraphs_from_lines(all_lines):
            story.append(Paragraph(para, normal_style))
            story.append(Spacer(1, 4 * mm))

    doc.build(story)


# ---------------- OPENAI PIPELINE ----------------

def call_buckeye_intake_single(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
) -> str:
    if project_description:
        user_content = (
            "PROJECT DESCRIPTION:\n"
            f"{project_description}\n\n"
            "FULL COMMERCIAL PLAN SET TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )
    else:
        user_content = (
            "FULL COMMERCIAL PLAN SET TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": BUCKEYE_INTAKE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def run_intake_review_pipeline(
    client: OpenAI,
    pdf_path: str,
    project_description: Optional[str],
) -> str:
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

    review_text = call_buckeye_intake_single(
        client,
        full_pdf_text=pdf_text,
        project_description=project_description,
    )

    return review_text


# ---------------- STREAMLIT UI ----------------

def main(embed: bool = False):
    """
    Commercial Plan Intake UI with session_state persistence,
    ICC button, and Reset button.
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – Commercial Plan Intake (Build 1.0)",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – Commercial Plan Intake (Build 1.0)")
    st.caption(f"Model: `{MODEL_NAME}`")

    st.write(
        "This tool helps determine whether a COMMERCIAL plan submittal is "
        "complete for intake. It checks required documents based on project "
        "type and whether the plans show food-service features."
    )

    st.info(
        "Step 1: Upload the full plan set PDF.\n"
        "Step 2: Add an optional project description.\n"
        "Step 3: Run the intake check to verify completeness."
    )

    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the app."
        )
        st.stop()

    client = get_client(env_api_key)

    if "intake_review" not in st.session_state:
        st.session_state["intake_review"] = ""
        st.session_state["intake_pdf_bytes"] = None
        st.session_state["intake_filename"] = ""

    uploaded_files = st.file_uploader(
        "Upload Commercial Plan Set PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    main_file = None
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        selected_name = st.selectbox(
            "Select which PDF to run the intake review on:",
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
            "Example: New 1-story retail shell building, Type IIB, unsprinklered. "
            "Includes new restrooms and site work; no commercial kitchen."
        ),
    )

    run_button = st.button("Run Intake Check", type="primary")

    if run_button:
        if not uploaded_files or main_file is None:
            st.error(
                "Please upload at least one PDF and select a primary plan set "
                "before running the intake review."
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
                "Step 1/2 – Extracting PDF text and sending for intake review ..."
            )

            review_text = run_intake_review_pipeline(
                client,
                tmp_path,
                project_description.strip() or None,
            )

            progress_bar.progress(80)
            status_placeholder.text(
                "Step 2/2 – Formatting intake table into PDF ..."
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix="_buckeye_intake_review.pdf",
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
            status_placeholder.text("Intake review complete.")

            st.session_state["intake_review"] = review_text
            st.session_state["intake_pdf_bytes"] = pdf_bytes
            st.session_state["intake_filename"] = main_file.name

        except Exception as e:
            st.error(f"Error during intake review: {e}")
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

        st.success("Intake review complete.")

    review_text = st.session_state.get("intake_review", "")
    pdf_bytes = st.session_state.get("intake_pdf_bytes", None)
    filename = st.session_state.get("intake_filename", "")

    if review_text:
        with st.expander("Show full AI intake report text"):
            st.text_area(
                "Intake report output",
                value=review_text,
                height=400,
            )

        if pdf_bytes:
            base_name = Path(filename).stem if filename else "intake"
            download_name = f"{base_name}_buckeye_intake_review.pdf"

            st.download_button(
                label="Download Intake Review PDF",
                data=pdf_bytes,
                file_name=download_name,
                mime="application/pdf",
                key="intake_pdf_download",
            )

        st.subheader("Reviewer Feedback (internal only)")
        st.write(
            "Use this section to rate the accuracy of this intake review and "
            "note any corrections. This does not change the model directly, "
            "but the data can be used to improve prompts and workflows."
        )

        rating = st.radio(
            "How accurate was this intake review?",
            ["Looks good", "Mostly okay", "Needs corrections"],
            index=0,
            key="intake_rating",
        )

        comments = st.text_area(
            "Notes / corrections",
            key="intake_comments",
            placeholder=(
                "Example: Energy compliance row should be 'Provided? – Yes' "
                "because COMcheck is on A0.1; missing row for grease interceptor."
            ),
        )

        if st.button("Save Intake Feedback", key="intake_save_feedback"):
            run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            csv_path = Path("feedback_intake.csv")

            extra_meta: Dict[str, Any] = {"model": MODEL_NAME}

            save_feedback_csv(
                csv_path=csv_path,
                tool_name="INTAKE",
                run_id=run_id,
                filename=filename or "(unknown)",
                rating=rating,
                comments=comments.strip(),
                extra=extra_meta,
            )

            st.success(
                f"Feedback saved to {csv_path.resolve().name} "
                f"in {csv_path.resolve().parent}."
            )

        st.markdown("---")

        col_icc, col_reset = st.columns(2)

        with col_icc:
            st.markdown(
                """
                <a href="https://codes.iccsafe.org/" target="_blank">
                    <button style="
                        background-color:#c45c26;
                        color:white;
                        border:none;
                        padding:0.35rem 1.1rem;
                        border-radius:999px;
                        font-weight:600;
                        cursor:pointer;
                    ">
                        Open ICC Codes (ICCsafe.org)
                    </button>
                </a>
                """,
                unsafe_allow_html=True,
            )

        with col_reset:
            if st.button("Start New Intake Review / Reset Results", key="intake_reset"):
                for k in ("intake_review", "intake_pdf_bytes", "intake_filename"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
    else:
        st.markdown(
            """
            <a href="https://codes.iccsafe.org/" target="_blank">
                <button style="
                    background-color:#c45c26;
                    color:white;
                    border:none;
                    padding:0.35rem 1.1rem;
                    border-radius:999px;
                    font-weight:600;
                    cursor:pointer;
                    margin-top:0.75rem;
                ">
                    Open ICC Codes (ICCsafe.org)
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
