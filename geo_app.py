import os
import io
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import csv
import datetime

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-5.1"  # OpenAI model for geotechnical review


# ---------------- PROMPT ----------------

BUCKEYE_GEOTECH_PROMPT = """
You are a City of Buckeye Building Safety Plans Examiner performing a
GEOTECHNICAL REVIEW for a building project in Arizona.

You are given the text of a geotechnical report (and possibly a short
project description from the reviewer). Use only the information in the
report itself and the project description. Do NOT invent data.

Your goal is to produce a SHORT, TABLE-ONLY SUMMARY capturing the key
design parameters for foundations and slabs. Focus on:

- Soil type / classification.
- Allowable bearing soil capacity for MAJOR structures (main buildings,
  primary structural elements).
- Allowable bearing soil capacity for MINOR structures (site walls, canopies,
  small accessory structures, sign foundations, etc.) if distinctly addressed.
- Recommended footing depth / embedment for MAJOR and MINOR structures.
- Coefficient of subgrade modulus (k-value) for slabs-on-grade or foundations,
  if provided.
- Any recommendations about the use of Portland cement (e.g., cement-treated
  subgrade, cement content, special mix or treatment), if provided.
- Any other concise, critical geotechnical recommendations that directly affect
  structural/foundation design.

If a specific value is not provided in the report, clearly state:
"Not provided in report."

If the report does not distinguish between major and minor structures, use
the same value in both columns or note "Same as major" or
"Not differentiated in report" as appropriate.

You MUST respond ONLY with a single GitHub-flavored Markdown table in this
exact format (no headings above or below the table, no narrative):

| Parameter | Major Structures | Minor Structures |

Where:
- "Parameter" is one of the following (at minimum):
    • Soil type / classification
    • Allowable bearing capacity
    • Footing depth / embedment
    • Subgrade modulus (k-value)
    • Portland cement recommendation
    • Other key geotechnical recommendations
- "Major Structures" column contains values for the main building or primary
  structures (with units, e.g., ksf, psf, pci, etc.).
- "Minor Structures" column contains values for minor/site structures, if
  distinct, or "Same as major" / "Not differentiated in report" as applicable.

Formatting requirements:
- Include the header row and separator row for a valid Markdown table.
- Use clear, concise text in each cell.
- Always show units exactly as written in the report (psf, ksf, pci, pcf, etc.).
- Do NOT fabricate any values. If not stated, write "Not provided in report."
- Do NOT include any text outside the table.
"""


# ---------------- HELPERS ----------------

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


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


def run_geotech_review(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
) -> Dict[str, Any]:
    """
    Call the OpenAI model with the geotech prompt and return the review text
    plus any usage information we can get.
    """
    # Truncate very large texts to avoid rate-limit issues
    MAX_CHARS = 80_000
    truncated_text = full_pdf_text
    if len(truncated_text) > MAX_CHARS:
        truncated_text = truncated_text[:MAX_CHARS]
        truncated_text += (
            "\n\n[NOTE TO REVIEWER: Geotechnical report text truncated to the first "
            f"{MAX_CHARS} characters to stay within model rate limits.]"
        )

    if project_description:
        user_content = (
            "PROJECT DESCRIPTION (from reviewer):\n"
            f"{project_description}\n\n"
            "GEOTECHNICAL REPORT TEXT EXTRACT:\n"
            f"{truncated_text}"
        )
    else:
        user_content = (
            "GEOTECHNICAL REPORT TEXT EXTRACT:\n"
            f"{truncated_text}"
        )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=BUCKEYE_GEOTECH_PROMPT,
        input=user_content,
    )

    review_text = response.output_text

    usage_summary: Dict[str, Any] = {}
    try:
        u = response.usage  # type: ignore[attr-defined]
        usage_summary = {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    except Exception:
        usage_summary = {}

    return {
        "review_text": review_text,
        "usage": usage_summary,
    }


def parse_markdown_table(md_table: str):
    """
    Parse a simple GitHub-flavored markdown table into a list-of-lists.

    Expects something like:
    | A | B | C |
    |---|---|---|
    | 1 | 2 | 3 |
    """
    lines = [ln.strip() for ln in md_table.splitlines() if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        # Skip separator row (---) by checking for all-dash cells
        if i == 1:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
        parts = [cell.strip() for cell in line.strip("|").split("|")]
        # Ensure exactly three columns
        if len(parts) < 3:
            parts += [""] * (3 - len(parts))
        elif len(parts) > 3:
            parts = parts[:3]
        rows.append(parts)
    return rows


def build_geotech_pdf(
    md_table: str,
    original_filename: str,
    logo_name: str = "City of Buckeye 2025.png",
) -> bytes:
    """
    Build a landscape PDF containing the Buckeye logo, title, and the
    geotechnical summary table.

    Column widths are fixed and cells are wrapped to keep everything readable.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is not installed. Install it with 'pip install reportlab'."
        )

    # Parse table
    raw_data = parse_markdown_table(md_table)
    if not raw_data:
        raw_data = [["Parameter", "Major Structures", "Minor Structures"],
                    ["No data", "No data", "No data"]]

    buffer = io.BytesIO()
    page_size = landscape(letter)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = []

    # Logo + title
    logo_path = Path(__file__).parent / logo_name
    if logo_path.exists():
        img = RLImage(str(logo_path))
        img.drawHeight = 50  # maintain aspect ratio
        img.drawWidth = 150
        story.append(img)
        story.append(Spacer(1, 12))

    title = "City of Buckeye – Geotechnical Summary"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 6))

    subtitle = f"Summary derived from geotechnical report: {original_filename}"
    story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 18))

    # Convert cells to Paragraphs for word wrapping
    body_style = ParagraphStyle(
        "GeoCell",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
    )

    data = []
    for r_idx, row in enumerate(raw_data):
        para_row = []
        for cell in row:
            # Replace newlines with spaces for nicer wrapping
            text = (cell or "").replace("\n", " ")
            para_row.append(Paragraph(text, body_style))
        data.append(para_row)

    # Determine column widths (3 columns) to fit page width
    total_width = page_size[0] - doc.leftMargin - doc.rightMargin
    col_widths = [
        total_width * 0.28,  # Parameter
        total_width * 0.36,  # Major
        total_width * 0.36,  # Minor
    ]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    orange = colors.HexColor("#c45c26")

    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), orange),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]
    )

    # Light row striping
    for row_idx in range(1, len(data)):
        if row_idx % 2 == 0:
            table_style.add("BACKGROUND", (0, row_idx), (-1, row_idx),
                            colors.whitesmoke)

    table.setStyle(table_style)
    story.append(table)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------- STREAMLIT UI ----------------

def main(embed: bool = False):
    """
    Geotechnical Review UI.

    embed = False → standalone (run this file directly).
    embed = True  → called from master tabbed app (no page_config).
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – Geotechnical Summary",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – Geotechnical Summary")
    st.caption(f"Model: `{MODEL_NAME}`")

    st.write(
        "Upload a geotechnical report PDF to generate a concise summary table of "
        "soil type, allowable bearing capacities, footing depths for major and "
        "minor structures, k-values (subgrade modulus), and any Portland cement "
        "recommendations directly affecting foundation design."
    )

    st.info(
        "This tool is intended as a summary aid for plan review. It does not "
        "replace a full engineering evaluation of the geotechnical report."
    )

    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the app."
        )
        st.stop()

    client = get_client(env_api_key)

    uploaded_file = st.file_uploader(
        "Upload Geotechnical Report (PDF)",
        type=["pdf"],
        accept_multiple_files=False,
    )

    project_description = st.text_area(
        "Optional project description",
        placeholder=(
            "Example: Retail shell building with slab-on-grade and shallow "
            "spread footings; minor structures include screen walls and "
            "site signage."
        ),
    )

    run_button = st.button("Run Geotechnical Summary", type="primary")

    if run_button:
        if not uploaded_file:
            st.error("Please upload a geotechnical report PDF before running the summary.")
            st.stop()

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        progress_bar.progress(5)
        status_placeholder.text("Uploading and preparing geotechnical report...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            progress_bar.progress(25)
            status_placeholder.text("Extracting text from PDF...")

            pdf_text = extract_pdf_text(tmp_path)
            if not pdf_text.strip():
                st.error("No extractable text found in the geotechnical report PDF.")
                return

            progress_bar.progress(50)
            status_placeholder.text("Calling OpenAI model for geotechnical summary...")

            result = run_geotech_review(
                client=client,
                full_pdf_text=pdf_text,
                project_description=project_description.strip() or None,
            )

            progress_bar.progress(100)
            status_placeholder.text("Geotechnical summary complete.")
        except Exception as e:
            st.error(f"Error during geotechnical summary: {e}")
            return
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        st.success("Geotechnical summary complete.")

        review_text = result.get("review_text", "").strip()
        usage_summary = result.get("usage", {}) or {}

        # Display the 3-column table as rendered Markdown
        st.subheader("Geotechnical Summary Table")
        st.markdown(review_text)

        # Optional: show raw text for copy/paste
        with st.expander("Show raw table text"):
            st.text_area(
                "Raw Markdown table",
                value=review_text,
                height=300,
            )

        base_name = Path(uploaded_file.name).stem if uploaded_file else "geotech_report"

        # Build and offer PDF if reportlab is available
        if review_text:
            if REPORTLAB_AVAILABLE:
                try:
                    pdf_bytes = build_geotech_pdf(
                        review_text,
                        original_filename=uploaded_file.name,
                    )
                    download_name_pdf = f"{base_name}_buckeye_geotech_summary.pdf"
                    st.download_button(
                        label="Download Summary Report (.pdf)",
                        data=pdf_bytes,
                        file_name=download_name_pdf,
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF summary: {e}")
            else:
                st.warning(
                    "PDF generation requires the 'reportlab' package. "
                    "Install it with `pip install reportlab` to enable PDF downloads."
                )

        # Show token usage if available
        if usage_summary:
            st.subheader("Token usage (if reported by API)")
            st.json(usage_summary)

        st.caption(
            "This tool summarizes key geotechnical design parameters for plan review. "
            "It does not replace the professional judgment of a registered engineer. "
            "License status of the geotechnical engineer of record must be verified "
            "directly in the Arizona Board of Technical Registration online register "
            "(https://azbtr.portalus.thentiacloud.net/webs/portal/register/#/)."
        )


if __name__ == "__main__":
    main()
