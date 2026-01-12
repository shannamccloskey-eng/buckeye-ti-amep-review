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

MODEL_INPUT_PRICE_PER_M = 5.00
MODEL_OUTPUT_PRICE_PER_M = 15.00


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

BUCKEYE_GEO_PROMPT = """
You are a City of Buckeye Building Safety Plans Examiner performing a
Geotechnical Summary Review for a building project.

You are given one or more geotechnical / soils reports as PDF text, and
optionally a brief project description.

Your task is to produce a concise, code-informed geotechnical summary that
can be used by plans examiners and structural engineers during building
review.

FOCUS AREAS
Summarize and clearly state, wherever the report provides them:

- Project location and basic site description.
- Soil profile and classifications (e.g., fill, native soils, expansions).
- Groundwater conditions and any seasonal variability.
- Recommended allowable bearing pressures (shallow foundations).
- Recommendations for deep foundations, if any.
- Modulus of subgrade reaction (if provided).
- Lateral earth pressures, sliding friction coefficients, etc.
- Seismic site class and site coefficients per ASCE 7-22, if discussed.
- Liquefaction / collapse / expansion potential, if discussed.
- Pavement design parameters, if applicable.
- Any recommendations for over-excavation, compaction, or fill placement.
- Any specific construction considerations (wet weather, utility trenches, etc.).
- Any notes about limitations, required updates, or need for additional borings.

OUTPUT FORMAT

1) PROJECT & SITE SUMMARY
Provide a short paragraph summarizing the site, soils, and intended project
(if known).

2) TABULAR SUMMARY OF KEY PARAMETERS
Provide a GitHub-flavored Markdown table with the following columns:

| Category | Parameter | Value / Range | Notes / Source |

Include rows for each important item above (bearing pressures, seismic site
class, etc.). If the report does not clearly state an item, mark the
Value/Range as "Not stated" and note that in the Notes column.

3) REVIEWER NOTES / FLAGS
Provide a short list of bullet points highlighting:
- Any unusually low or high values.
- Any conditional recommendations (e.g., depends on final footing width,
  depends on slab loading, etc.).
- Any limitations (e.g., report based on preliminary plans, recommends
  updated borings if building location changes, etc.).
- Any items that should be cross-checked against the structural design.

CODES / REFERENCES
When you mention code references, use only:
- 2024 International Building Code (IBC)
- 2024 International Residential Code (IRC) if applicable
- ASCE 7-22
- City of Buckeye Amendments

Do NOT quote code text; only reference section numbers or standard names.

Your tone should be concise, technical, and suitable for inclusion in a
plans examiner's geotechnical summary file.
"""


# ---------------- PDF HELPERS ----------------

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
    header_prefix: str = "| Category",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Find the Markdown table starting with the geotech header
    '| Category | Parameter | Value / Range | Notes / Source |'.
    Returns (pre_lines, table_lines, post_lines).
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
    Convert GitHub-flavored Markdown table lines to lists of strings.
    Assumes first line is header and second line is separator.
    """
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
    """
    Landscape PDF with Buckeye logo, title, and a formatted table if the
    AI output contains the geotech Markdown table:

    | Category | Parameter | Value / Range | Notes / Source |
    """
    stylesheet = getSampleStyleSheet()
    normal_style = stylesheet["Normal"]

    all_lines = text.splitlines()
    _, table_lines, post_lines = _extract_markdown_table(all_lines)
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

    title = "City of Buckeye – Geotechnical Summary Review"
    story.append(Paragraph(title, stylesheet["Title"]))
    story.append(Spacer(1, 3 * mm))

    subtitle = f"Summary generated from geotechnical report: {original_filename}"
    story.append(Paragraph(subtitle, normal_style))
    story.append(Spacer(1, 8 * mm))

    if table_data_raw:
        # Build a nicely formatted table, similar to intake_app
        header_row = [
            "Category",
            "Parameter",
            "Value / Range",
            "Notes / Source",
        ]

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
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            elif len(row) > 4:
                row = row[:4]
            full_table_data.append([make_cell(c) for c in row])

        total_width = pagesize[0] - doc.leftMargin - doc.rightMargin
        # Reasonable widths for the four geotech columns
        col_widths = [
            total_width * 0.20,  # Category
            total_width * 0.28,  # Parameter
            total_width * 0.18,  # Value / Range
            total_width * 0.34,  # Notes / Source
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

        # Alternating row shading
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
        story.append(Spacer(1, 6 * mm))

        # Add any post-table narrative (project summary, reviewer notes, etc.)
        post_paragraphs = _split_paragraphs_from_lines(post_lines)
        for para in post_paragraphs:
            story.append(Paragraph(para, normal_style))
            story.append(Spacer(1, 3 * mm))
    else:
        # Fallback – no table detected, just dump text in paragraphs
        for para in _split_paragraphs_from_lines(all_lines):
            story.append(Paragraph(para, normal_style))
            story.append(Spacer(1, 4 * mm))

    doc.build(story)


# ---------------- OPENAI PIPELINE ----------------

def call_buckeye_geo_single(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    if project_description:
        user_content = (
            "PROJECT DESCRIPTION:\n"
            f"{project_description}\n\n"
            "FULL GEOTECHNICAL REPORT TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )
    else:
        user_content = (
            "FULL GEOTECHNICAL REPORT TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": BUCKEYE_GEO_PROMPT},
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
) -> Tuple[str, Dict[str, Any]]:
    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text.strip():
        raise ValueError("No extractable text found in PDF.")

    MAX_CHARS = 80_000
    if len(pdf_text) > MAX_CHARS:
        pdf_text = pdf_text[:MAX_CHARS]
        pdf_text += (
            "\n\n[NOTE TO REVIEWER: Geotechnical text truncated to the first "
            f"{MAX_CHARS} characters to stay within model rate limits.]"
        )

    review_text, usage = call_buckeye_geo_single(
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

    cost_input = (input_tokens / 1_000_000.0) * MODEL_INPUT_PRICE_PER_M
    cost_output = (output_tokens / 1_000_000.0) * MODEL_OUTPUT_PRICE_PER_M
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
    Geotechnical Summary Review UI with:
    - multi-file upload
    - blue/bold manifest preview
    - reset that also clears uploads
    - download + reset on the same row
    - internal token / feedback section
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – Geotechnical Summary Review (Beta)",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – Geotechnical Summary Review (Beta)")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.write(
        "Upload geotechnical / soils report PDF(s) to generate a structured "
        "geotechnical summary for use in building and structural plan review."
    )

    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the app."
        )
        st.stop()

    client = get_client(env_api_key)

    # ---------- uploader key for full reset ----------
    if "geo_uploader_key" not in st.session_state:
        st.session_state["geo_uploader_key"] = 0

    # ---------- Handle reset trigger BEFORE widgets are created ----------
    if st.session_state.get("geo_reset_trigger", False):
        for k in ("geo_review", "geo_usage", "geo_pdf_bytes", "geo_filename"):
            st.session_state.pop(k, None)

        st.session_state["geo_uploader_key"] += 1  # clears uploader
        st.session_state["geo_reset_trigger"] = False

    # ---------- Initialize session defaults ----------
    if "geo_review" not in st.session_state:
        st.session_state["geo_review"] = ""
        st.session_state["geo_usage"] = {}
        st.session_state["geo_pdf_bytes"] = None
        st.session_state["geo_filename"] = ""

    uploader_key = f"geo_files_{st.session_state['geo_uploader_key']}"

    uploaded_files = st.file_uploader(
        "Upload Geotechnical / Soils Report PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    main_file = None
    file_names: List[str] = []

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        selected_name = st.selectbox(
            "Select which PDF to run the geotechnical summary on:",
            file_names,
            index=0,
        )
        for f in uploaded_files:
            if f.name == selected_name:
                main_file = f
                break

        st.markdown("**Uploaded files (manifest preview):**")
        for name in file_names:
            # Blue & bold to match the rest of the tools
            st.markdown(
                f"<span style='color:#1d4ed8; font-weight:600'>• {name}</span>",
                unsafe_allow_html=True,
            )

    project_description = st.text_area(
        "Optional project description / notes",
        placeholder=(
            "Example: New single-story restaurant building with shallow "
            "spread footings; one-story CMU and steel framing; "
            "consider potential for expansion and seismic site class."
        ),
    )

    run_button = st.button("Run Geotechnical Summary Review", type="primary")

    if run_button:
        if not uploaded_files or main_file is None:
            st.error(
                "Please upload at least one geotechnical PDF and select a primary "
                "report before running the summary review."
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
                "Step 1/3 – Extracting geotechnical text and sending for summary ..."
            )

            review_text, usage_summary = run_review_pipeline_single(
                client,
                tmp_path,
                project_description.strip() or None,
            )

            progress_bar.progress(70)
            status_placeholder.text(
                "Step 2/3 – Formatting geotechnical summary into PDF ..."
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix="_buckeye_geo_summary.pdf",
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
            status_placeholder.text("Step 3/3 – Geotechnical summary complete.")

            st.session_state["geo_review"] = review_text
            st.session_state["geo_usage"] = usage_summary
            st.session_state["geo_pdf_bytes"] = pdf_bytes
            st.session_state["geo_filename"] = main_file.name

        except Exception as e:
            st.error(f"Error during geotechnical summary review: {e}")
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

        st.success("Geotechnical summary review complete.")

    review_text = st.session_state.get("geo_review", "")
    usage_summary = st.session_state.get("geo_usage", {}) or {}
    pdf_bytes = st.session_state.get("geo_pdf_bytes", None)
    filename = st.session_state.get("geo_filename", "")

    if review_text:
        with st.expander("Show full geotechnical summary text"):
            st.text_area(
                "Summary output",
                value=review_text,
                height=400,
            )

        # Download + Reset buttons on same row (like intake & TI AMEP)
        button_cols = st.columns([2, 1])
        with button_cols[0]:
            if pdf_bytes:
                base_name = Path(filename).stem if filename else "geo_summary"
                download_name = f"{base_name}_buckeye_geo_summary.pdf"

                st.download_button(
                    label="Download Geotechnical Summary PDF",
                    data=pdf_bytes,
                    file_name=download_name,
                    mime="application/pdf",
                    key="geo_pdf_download",
                )

        with button_cols[1]:
            if st.button("Start New Geotech Review / Reset Results", key="geo_reset"):
                st.session_state["geo_reset_trigger"] = True
                st.rerun()

        # Internal section separator
        st.markdown("---")

        if usage_summary:
            st.subheader("Token usage & cost estimate")
            st.json(usage_summary)

            if "cost_total_usd" in usage_summary:
                st.markdown(
                    f"**Estimated API cost ({MODEL_NAME}): "
                    f"${usage_summary['cost_total_usd']:.4f} USD** "
                    f"(input: ${usage_summary['cost_input_usd']:.4f}, "
                    f"output: ${usage_summary['cost_output_usd']:.4f})"
                )

        st.subheader("Reviewer Feedback (internal only)")
        st.write(
            "Use this section to rate the accuracy of this geotechnical summary "
            "and note any corrections. This does not change the model directly, "
            "but the data can be used to improve prompts and workflows."
        )

        rating = st.radio(
            "How accurate was this geotechnical summary?",
            ["Looks good", "Mostly okay", "Needs corrections"],
            index=0,
            key="geo_rating",
        )

        comments = st.text_area(
            "Notes / corrections",
            key="geo_comments",
            placeholder=(
                "Example: Recommended bearing pressure appears lower than used in "
                "structural design; confirm with engineer. "
                "Site class stated as D, not C."
            ),
        )

        if st.button("Save Geotechnical Feedback", key="geo_save_feedback"):
            run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            csv_path = Path("feedback_geo_summary.csv")

            extra_meta = {"model": MODEL_NAME}
            extra_meta.update(usage_summary or {})

            save_feedback_csv(
                csv_path=csv_path,
                tool_name="GEO_SUMMARY",
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
    else:
        # No summary yet
        pass


if __name__ == "__main__":
    main()
