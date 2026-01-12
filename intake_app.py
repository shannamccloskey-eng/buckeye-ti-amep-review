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
# IMPORTANT: final output MUST include a single GitHub table with:
# | Item | Required? | Provided? | Comments / Required Action |

BUCKEYE_COMMERCIAL_INTAKE_PROMPT = """
You are a City of Buckeye Building Safety Plans Examiner performing a
Commercial Plan Intake review for a building permit submittal.

You are given:
- A list of uploaded PDF filenames (the "manifest").
- Extracted text from the PRIMARY plan set PDF (usually architectural / combined).
- An optional project description typed by the user.

Your job is to:
1. Identify what documents are present in the manifest.
2. Determine, based on the plan text and description, what documents SHOULD be
   included for a complete commercial building permit submittal.
3. Identify any missing or obviously incomplete items as intake discrepancies.
4. Summarize everything as a clean intake checklist in table form.

Only reference the following codes / standards by exact name and year:
- 2024 International Building Code (IBC)
- 2024 International Mechanical Code (IMC)
- 2024 International Plumbing Code (IPC)
- 2024 International Fire Code (IFC)
- 2017 ICC A117.1
- 2023 National Electrical Code (NEC, NFPA 70)
- 2018 International Energy Conservation Code (IECC)
- ASCE 7-22
- City of Buckeye Amendments

Do NOT quote code text; only reference section numbers / titles.

------------------------------------------------------------
DOCUMENT CLASSIFICATION – MANIFEST
------------------------------------------------------------

From the filenames in the manifest, classify documents into logical buckets such as:
- Application / cover sheet
- Architectural / civil / site / demo / life safety plans
- Structural plans
- Structural calculations
- Geotechnical / soils report (include "geo", "GEO" variants)
- COMcheck / energy compliance (any discipline: ARCH, MECH, ELEC)
- Mechanical plans
- Electrical plans
- Plumbing plans
- Fire protection plans (sprinkler, fire alarm, hood suppression if clearly named)
- Special inspection certificates / agreements (include "SIC", "sic", "SPECIAL INSP")
- Truss / delegated design / deferred submittals
- Other supporting documents (specs, reports, etc.)

Be tolerant of naming variations and abbreviations:
- Structural calcs: "STRUCTURAL CALCS", "STR CALCS", "STR CALC", etc.
- Geotech: "geo", "geotech", "soils", "GEO", etc.
- Special inspections: "SPECIAL INSP CERT", "SPECIAL INSP", "SIC", "S.I.C." etc.
- COMcheck: any filename containing "COM CHECK", "COMCHECK", etc.

------------------------------------------------------------
SPECIAL SCOPE TRIGGERS – COMMERCIAL KITCHEN & WALK-INS
------------------------------------------------------------

From BOTH:
- The plan text extract, AND
- The manifest filenames,

determine whether the project includes the following scopes. Remember that
supporting information may be:
- in the primary plan set (mechanical sheets, hood details, walk-in slab
  details, etc.), OR
- in separate PDFs (cut sheets, manuals, separate hood drawings, etc.).

1) COMMERCIAL KITCHEN / COOKING EQUIPMENT
   Look for words / phrases such as: commercial kitchen, restaurant, food service,
   cooking appliances, ranges, ovens, fryers, griddles, Type I hood, Type II hood,
   grease duct, exhaust hood, makeup air, or similar.

   If commercial cooking / kitchen work is present, then:
   - Determine whether hood-related information appears to be provided EITHER:
     * in the mechanical / architectural sheets of the main plan set, OR
     * as separate hood / mechanical / specification PDFs in the manifest.
   - Required hood-related content includes, as applicable:
     * Kitchen hood(s) shown on mechanical / architectural plans.
     * Hood schedules or hood specifications.
     * Grease duct routing and termination details.
     * Makeup air provisions.
     * Indication of hood fire suppression (even if under separate permit).

   If this information is clearly present in the main plan set or a separate
   document, treat it as PROVIDED.
   If it is not clearly present in either place, treat it as REQUIRED BUT NOT
   PROVIDED and flag it in the checklist.

2) WALK-IN COOLER / FREEZER / REFRIGERATOR
   Look for words / phrases such as: walk-in cooler, walk-in freezer,
   walk-in refrigerator, "walk in cooler", "walk-in box", refrigerated box, etc.

   If a walk-in cooler / freezer / refrigerator is present, then:
   - Determine whether supporting information appears to be provided EITHER:
     * in the main plan set (slab details, insulation / vapor barrier, structural
       support, roof/ceiling framing, etc.), OR
     * in separate PDFs (manufacturer manuals / cut sheets, structural review, etc.).

   Supporting information may include:
     * Manufacturer installation manual / cut sheets for the walk-in unit.
     * Foundation / slab details showing insulation and vapor barrier, if applicable.
     * Structural support or roof framing review if the unit is roof-mounted.
     * Refrigeration equipment data or schedule if clearly required by scope.

   If this information is clearly present in the main plan set or a separate
   document, treat it as PROVIDED.
   If it is not clearly present in either place, treat it as REQUIRED BUT NOT
   PROVIDED and flag it in the checklist.

------------------------------------------------------------
INTAKE CHECKLIST TABLE FORMAT (CRITICAL)
------------------------------------------------------------

Your final output MUST include a single GitHub-flavored Markdown table
with EXACTLY these 4 columns, in this exact order and spelling:

| Item | Required? | Provided? | Comments / Required Action |

The "Required?" column should typically be "Yes", "No", or "Maybe"
(for rare edge cases).

The "Provided?" column should state whether the required document or
information is clearly present, e.g.:
- "Yes – in plan set"
- "Yes – see separate PDF (filename)"
- "No"
- "Deferred" (for deferred submittals like trusses, hood suppression, etc.)

The "Comments / Required Action" column should:
- Briefly explain the status.
- If something is missing or incomplete, clearly state what the applicant
  needs to provide at intake.

You should include rows for typical commercial intake items such as:
- Application / cover sheet
- Code analysis (occupancy, construction type, allowable area, fire protection)
- Site / civil plan
- Architectural floor plans & RCP
- Elevations / sections
- Structural plans
- Structural calculations
- Geotechnical / soils report
- Mechanical plans / schedules
- Electrical plans / lighting
- Plumbing plans / isometrics
- Energy compliance / COMcheck
- Special inspections
- Truss / delegated design / deferred submittals
- Any special-scope items, including:
  * Kitchen hood / commercial kitchen documentation (if applicable)
  * Walk-in cooler / freezer / refrigerator documentation (if applicable)
  * Any other scope-specific documents clearly required by the described work.

You may also include "Not applicable" or "N/A" items if you want to show
that a category was considered but not required.

STRUCTURAL RULES:

- The header row MUST be:

  | Item | Required? | Provided? | Comments / Required Action |

- The second row MUST be the separator, for example:

  | --- | --- | --- | --- |

- Each subsequent row MUST:
  - Start with `|` and end with `|`.
  - Have exactly 4 columns separated by `|`.

- Do NOT wrap the table in code fences (no ```).
- Do NOT put any extra narrative text inside or between the table rows.

You may include a short narrative summary *before* the table
(e.g., headings, bullets, notes), but the table itself must strictly
follow the format above so it can be parsed and turned into a PDF
checklist.
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
    header_prefix: str = "| Item | Required?",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Find the Markdown intake checklist table starting with the header
    '| Item | Required? | Provided? | Comments / Required Action |'.
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
    Landscape PDF with Buckeye logo, title, and a formatted intake checklist
    table styled like the previous sample:
    - orange header band
    - 4 columns: Item / Required? / Provided? / Comments / Required Action
    - grid lines and consistent padding
    """
    stylesheet = getSampleStyleSheet()
    normal_style = stylesheet["Normal"]

    all_lines = text.splitlines()
    pre_lines, table_lines, post_lines = _extract_markdown_table(all_lines)
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

    # Logo
    logo_path = Path(__file__).parent / logo_name
    if logo_path.exists():
        img = RLImage(str(logo_path))
        img.drawHeight = 50
        img.drawWidth = 150
        story.append(img)
        story.append(Spacer(1, 8 * mm))

    # Title
    title = "City of Buckeye – Commercial Plan Intake"
    story.append(Paragraph(title, stylesheet["Title"]))
    story.append(Spacer(1, 3 * mm))

    subtitle = f"Intake review generated from plan set: {original_filename}"
    story.append(Paragraph(subtitle, normal_style))
    story.append(Spacer(1, 8 * mm))

    # (Optional) pre-table narrative if we ever want it – keep it minimal
    for para in _split_paragraphs_from_lines(pre_lines):
        story.append(Paragraph(para, normal_style))
        story.append(Spacer(1, 3 * mm))

    # Intake checklist table
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
            fontSize=9,
            leading=11,
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
        # Approximate the proportions from your sample:
        # Item (wide-ish), Required (narrow), Provided (medium), Comments (widest)
        col_widths = [
            total_width * 0.23,  # Item
            total_width * 0.08,  # Required?
            total_width * 0.14,  # Provided?
            total_width * 0.55,  # Comments / Required Action
        ]

        table = Table(full_table_data, colWidths=col_widths, repeatRows=1)

        header_color = colors.HexColor("#c45c26")

        table_style = TableStyle(
            [
                # Header
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 0), (-1, 0), 5),

                # Body
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 1), (-1, -1), "TOP"),
                ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 1), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 3),

                # Grid
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )

        # Zebra shading on body rows (optional, subtle)
        for r_idx in range(1, len(full_table_data)):
            if r_idx % 2 == 0:
                table_style.add(
                    "BACKGROUND", (0, r_idx), (-1, r_idx), colors.whitesmoke
                )

        table.setStyle(table_style)

        story.append(table)
        story.append(Spacer(1, 6 * mm))

    # Optional post-table narrative (rarely used)
    for para in _split_paragraphs_from_lines(post_lines):
        story.append(Paragraph(para, normal_style))
        story.append(Spacer(1, 3 * mm))

    doc.build(story)


# ---------------- OPENAI PIPELINE ----------------

def call_buckeye_intake_single(
    client: OpenAI,
    full_pdf_text: str,
    manifest_files: List[str],
    project_description: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    manifest_block = "FILE MANIFEST (all uploaded PDFs):\n" + "\n".join(
        f"- {name}" for name in manifest_files
    )

    if project_description:
        user_content = (
            "PROJECT DESCRIPTION:\n"
            f"{project_description}\n\n"
            f"{manifest_block}\n\n"
            "PRIMARY PLAN SET TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )
    else:
        user_content = (
            f"{manifest_block}\n\n"
            "PRIMARY PLAN SET TEXT EXTRACT:\n"
            f"{full_pdf_text}"
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": BUCKEYE_COMMERCIAL_INTAKE_PROMPT},
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
    manifest_files: List[str],
    project_description: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
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

    review_text, usage = call_buckeye_intake_single(
        client,
        full_pdf_text=pdf_text,
        manifest_files=manifest_files,
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
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – Commercial Plan Intake (Beta)",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – Commercial Plan Intake (Beta)")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.write(
        "Upload the complete commercial permit submittal (all PDFs) to generate "
        "a plan intake checklist in the standard City of Buckeye format, including "
        "checks for structural calculations, geotechnical report, COMcheck, special "
        "inspections, commercial kitchen hood documentation, walk-in cooler/freezer "
        "information, and other supporting documents."
    )

    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the app."
        )
        st.stop()

    client = get_client(env_api_key)

    # uploader key for full reset
    if "intake_uploader_key" not in st.session_state:
        st.session_state["intake_uploader_key"] = 0

    # handle reset trigger before widgets
    if st.session_state.get("intake_reset_trigger", False):
        for k in (
            "intake_review",
            "intake_usage",
            "intake_pdf_bytes",
            "intake_filename",
            "intake_manifest_snapshot",
        ):
            st.session_state.pop(k, None)

        st.session_state["intake_uploader_key"] += 1
        st.session_state["intake_reset_trigger"] = False

    # defaults
    if "intake_review" not in st.session_state:
        st.session_state["intake_review"] = ""
        st.session_state["intake_usage"] = {}
        st.session_state["intake_pdf_bytes"] = None
        st.session_state["intake_filename"] = ""
        st.session_state["intake_manifest_snapshot"] = []

    uploader_key = f"intake_files_{st.session_state['intake_uploader_key']}"

    uploaded_files = st.file_uploader(
        "Upload entire commercial submittal set (PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    main_file = None
    file_names: List[str] = []

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]

        st.markdown(
            "Select the PRIMARY plan set PDF for text extraction "
            "(usually the main architectural or combined set):"
        )
        selected_name = st.selectbox(
            "",
            file_names,
            index=0,
        )
        for f in uploaded_files:
            if f.name == selected_name:
                main_file = f
                break

        st.markdown("**Uploaded files (manifest preview):**")
        for name in file_names:
            st.markdown(
                f"<span style='color:#1d4ed8; font-weight:600'>• {name}</span>",
                unsafe_allow_html=True,
            )

    project_description = st.text_area(
        "Optional project description / notes",
        placeholder=(
            "Example: New shell restaurant building with site work; "
            "non-separated B occupancy with A-2 dining; Type IIB; fire sprinklered."
        ),
    )

    run_button = st.button("Run Commercial Plan Intake Review", type="primary")

    if run_button:
        if not uploaded_files or main_file is None:
            st.error(
                "Please upload at least one PDF and select a PRIMARY plan set "
                "before running the intake review."
            )
            st.stop()

        manifest_snapshot = [f.name for f in uploaded_files]

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
                "Step 1/3 – Extracting PRIMARY plan text and sending for intake review ..."
            )

            review_text, usage_summary = run_review_pipeline_single(
                client,
                tmp_path,
                manifest_files=manifest_snapshot,
                project_description=project_description.strip() or None,
            )

            progress_bar.progress(70)
            status_placeholder.text(
                "Step 2/3 – Formatting intake checklist into PDF ..."
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
            status_placeholder.text("Step 3/3 – Commercial intake review complete.")

            st.session_state["intake_review"] = review_text
            st.session_state["intake_usage"] = usage_summary
            st.session_state["intake_pdf_bytes"] = pdf_bytes
            st.session_state["intake_filename"] = main_file.name
            st.session_state["intake_manifest_snapshot"] = manifest_snapshot

        except Exception as e:
            st.error(f"Error during Commercial intake review: {e}")
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

        st.success("Commercial intake review complete.")

    review_text = st.session_state.get("intake_review", "")
    usage_summary = st.session_state.get("intake_usage", {}) or {}
    pdf_bytes = st.session_state.get("intake_pdf_bytes", None)
    filename = st.session_state.get("intake_filename", "")
    manifest_snapshot = st.session_state.get("intake_manifest_snapshot", [])

    if review_text:
        # NOTE: we intentionally removed the "Show full AI intake review text"
        # expander per request. Only PDF + metadata remain in the UI.

        # Download + Reset row
        button_cols = st.columns([2, 1])
        with button_cols[0]:
            if pdf_bytes:
                base_name = Path(filename).stem if filename else "commercial_intake"
                download_name = f"{base_name}_buckeye_intake_review.pdf"

                st.download_button(
                    label="Download Commercial Intake Review PDF",
                    data=pdf_bytes,
                    file_name=download_name,
                    mime="application/pdf",
                    key="intake_pdf_download",
                )

        with button_cols[1]:
            if st.button(
                "Start New Commercial Intake / Reset Results",
                key="intake_reset",
            ):
                st.session_state["intake_reset_trigger"] = True
                st.rerun()

        # Manifest used
        if manifest_snapshot:
            st.markdown("**File manifest used for this intake:**")
            for name in manifest_snapshot:
                st.markdown(
                    f"<span style='color:#1f2933; font-weight:600'>• {name}</span>",
                    unsafe_allow_html=True,
                )

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
            "Use this section to rate the accuracy of this commercial intake review "
            "and note any corrections. This does not change the model directly, "
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
                "Example: Structural calcs actually present in combined PDF; "
                "hood schedule located on M3.1; walk-in cooler slab detail is S5."
            ),
        )

        if st.button("Save Commercial Intake Feedback", key="intake_save_feedback"):
            run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            csv_path = Path("feedback_commercial_intake.csv")

            extra_meta = {"model": MODEL_NAME}
            extra_meta.update(usage_summary or {})

            save_feedback_csv(
                csv_path=csv_path,
                tool_name="COMM_INTAKE",
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


if __name__ == "__main__":
    main()
