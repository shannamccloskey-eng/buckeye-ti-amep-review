import os
import csv
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tempfile
from functools import lru_cache

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm


# ---------------- CONFIG ----------------

# OpenAI model configuration
MODEL_NAME = "gpt-4.1"
MODEL_INPUT_PRICE_PER_M = 5.00
MODEL_OUTPUT_PRICE_PER_M = 15.00

# Folder named "IBC 2024" sitting next to this file:
#   /Users/sm/Documents/buckeye-ti-amep-review/
#       app.py
#       ti_amep_app.py
#       IBC 2024/   <-- your IBC PDFs
BASE_DIR = Path(__file__).resolve().parent
CODE_PDF_DIR = BASE_DIR / "IBC 2024"


def get_client(api_key: str) -> OpenAI:
    """Create an OpenAI client using the provided API key."""
    return OpenAI(api_key=api_key)


# ---------------- PROMPT ----------------

BUCKEYE_TI_AMEP_PROMPT = """
You are a City of Buckeye Building Safety Plans Examiner performing an
Architectural, Mechanical, Electrical, Plumbing (AMEP) plan review for a
Tenant Improvement (TI) project within an existing building in the City of Buckeye.

Perform a complete AMEP review of the submitted TI plans using the adopted codes:

- 2024 International Building Code (IBC)
- 2024 International Mechanical Code (IMC)
- 2024 International Plumbing Code (IPC)
- 2024 International Fire Code (IFC)
- 2017 ICC A117.1
- 2023 National Electrical Code (NEC / NFPA 70)
- 2023 International Energy Conservation Code (IECC)
- 2010 ADA Standards for Accessible Design
- City of Buckeye Amendments

SCOPE OF REVIEW
Classify the project and review it under the following headings:
- Architectural
- Mechanical
- Electrical
- Plumbing
- Fire & Life Safety
- Accessibility
- Energy

DELIVERABLES
Provide your response in three clearly labeled sections:

1. PROJECT SCOPE SUMMARY

2. DISCREPANCY TABLE
   - Present as a Markdown table with the following exact column headers:

     | Sheet Reference | Discipline | Description of Issue | Code Section | Required Correction |

   - Each row must identify sheet, discipline, issue, code section, and required correction.
   - If no discrepancies are found, provide a single-row table stating
     "No discrepancies noted" in the Description of Issue column.

3. ENERGOV-COMPATIBLE COMMENTS
   - Format: [Discipline] – [Code Reference]: [Required Correction].

FINAL SUMMARY & DETERMINATION
Conclude with:
- Overall assessment of the TI design.
- One of: APPROVED, APPROVED WITH COMMENTS, or CORRECTIONS REQUIRED (in ALL CAPS).
- Closing statement noting review performed in accordance with adopted codes and City amendments.

ADDITIONAL INSTRUCTIONS
- Base all code citations primarily on the adopted-code text provided in context.
- Do not quote long passages of code. Cite sections and summarize requirements.
- If plans omit information required for a full review, note as discrepancies and request missing info.
"""


# ---------------- PDF / TEXT HELPERS ----------------

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using pypdf (text-based PDFs only)."""
    reader = PdfReader(pdf_path)
    all_text: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            all_text.append(page_text)
    return "\n\n".join(all_text).strip()


@lru_cache(maxsize=1)
def load_code_context(max_chars: int = 40_000) -> str:
    """
    Load and concatenate adopted-code text (e.g., IBC 2024) from CODE_PDF_DIR.

    Truncates to max_chars to stay within the model context window.
    Returns "" if nothing readable is found.
    """
    if not CODE_PDF_DIR.exists() or not CODE_PDF_DIR.is_dir():
        return ""

    chunks: List[str] = []
    for entry in sorted(CODE_PDF_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".pdf":
            try:
                txt = extract_pdf_text(str(entry))
                if txt.strip():
                    header = f"--- CODE FILE: {entry.name} ---\n"
                    chunks.append(header + txt)
            except Exception:
                continue

    full = "\n\n".join(chunks).strip()
    if not full:
        return ""

    if len(full) > max_chars:
        full = (
            full[:max_chars]
            + "\n\n[NOTE: Code PDF context truncated for length; "
              "only a portion of the adopted-code text is included.]"
        )
    return full


def _clean_markdown_for_pdf(text: str) -> str:
    """
    Very lightweight markdown cleanup for PDF output:
    - strip **bold** markers
    - remove trailing '---' rule lines
    """
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if line.strip() in ("---", "___"):
            cleaned_lines.append("")
            continue
        cleaned_lines.append(line.replace("**", ""))
    return "\n".join(cleaned_lines)


def create_review_pdf(
    review_text: str,
    usage_summary: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Create a simple PDF summary of the review text plus a Run Log section
    using ReportLab.
    """
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    heading_style = styles["Heading1"]

    body_style = ParagraphStyle(
        "Body",
        parent=normal_style,
        fontSize=9,
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    story: List[Any] = []

    # Header
    story.append(Paragraph("City of Buckeye – TI AMEP Review", heading_style))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            datetime.datetime.now().strftime("Generated on %Y-%m-%d %H:%M"),
            normal_style,
        )
    )
    story.append(Spacer(1, 12))

    # Main review body (cleaned markdown -> plain text with line breaks)
    body_text = _clean_markdown_for_pdf(review_text)
    safe_text = (
        body_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    story.append(Paragraph(safe_text, body_style))

    # Run log section
    if usage_summary:
        story.append(Spacer(1, 18))
        story.append(Paragraph("Run Log", styles["Heading2"]))
        story.append(Spacer(1, 6))

        # Build a small log as HTML-ish text list
        log_lines: List[str] = []

        input_tokens = usage_summary.get("input_tokens", 0)
        output_tokens = usage_summary.get("output_tokens", 0)
        reasoning_tokens = usage_summary.get("reasoning_tokens", 0)
        total_tokens = usage_summary.get("total_tokens", 0)
        model = usage_summary.get("model", MODEL_NAME)
        cost_input = usage_summary.get("cost_input_usd", 0.0)
        cost_output = usage_summary.get("cost_output_usd", 0.0)
        cost_total = usage_summary.get("cost_total_usd", 0.0)

        log_lines.append(f"Model: {model}")
        log_lines.append(f"Input tokens: {input_tokens:,}")
        log_lines.append(f"Output tokens: {output_tokens:,}")
        if reasoning_tokens:
            log_lines.append(f"Reasoning tokens: {reasoning_tokens:,}")
        log_lines.append(f"Total tokens: {total_tokens:,}")
        log_lines.append(
            f"Estimated cost (input): ${cost_input:0.4f} USD "
            f"(at ${MODEL_INPUT_PRICE_PER_M}/M tokens)"
        )
        log_lines.append(
            f"Estimated cost (output): ${cost_output:0.4f} USD "
            f"(at ${MODEL_OUTPUT_PRICE_PER_M}/M tokens)"
        )
        log_lines.append(f"Estimated total cost: ${cost_total:0.4f} USD")

        log_html = "<br/>".join(log_lines)
        story.append(Paragraph(log_html, body_style))

    doc.build(story)


# ---------------- OPENAI CALL ----------------

def call_buckeye_ti_amep_single(
    client: OpenAI,
    full_pdf_text: str,
    project_description: Optional[str],
    code_context: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Call the OpenAI model with:
    - System prompt describing the TI AMEP review role
    - Optional system message containing adopted-code text
    - User message containing project description + TI plan text
    """
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

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": BUCKEYE_TI_AMEP_PROMPT},
    ]

    if code_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "The following text is drawn from the adopted codes "
                    "(including IBC 2024 and related standards) and should be "
                    "used as the primary basis for code section citations. "
                    "Do NOT quote long passages verbatim; instead, cite and "
                    "interpret the sections.\n\n"
                    f"{code_context}"
                ),
            }
        )

    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
    )

    content = response.choices[0].message.content or ""
    usage_data: Dict[str, Any] = {
        "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
    }
    reasoning_tokens = getattr(response.usage, "reasoning_tokens", 0) or 0
    usage_data["reasoning_tokens"] = reasoning_tokens

    return content, usage_data


def run_review_pipeline_single(
    client: OpenAI,
    pdf_path: Path,
    project_description: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from the TI plan PDF, load code context,
    and call the OpenAI TI AMEP reviewer.
    """
    pdf_text = extract_pdf_text(str(pdf_path))
    if not pdf_text.strip():
        raise ValueError("No extractable text found in PDF.")

    MAX_CHARS = 80_000
    if len(pdf_text) > MAX_CHARS:
        pdf_text = (
            pdf_text[:MAX_CHARS]
            + "\n\n[NOTE TO REVIEWER: Plan text truncated to the first "
              f"{MAX_CHARS} characters to stay within model limits.]"
        )

    code_context = load_code_context()
    review_text, usage = call_buckeye_ti_amep_single(
        client=client,
        full_pdf_text=pdf_text,
        project_description=project_description,
        code_context=code_context,
    )

    # Cost summary
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens + reasoning_tokens)

    cost_input = (input_tokens / 1_000_000.0) * MODEL_INPUT_PRICE_PER_M
    cost_output = (output_tokens / 1_000_000.0) * MODEL_OUTPUT_PRICE_PER_M
    cost_total = cost_input + cost_output

    usage_summary: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_input_usd": cost_input,
        "cost_output_usd": cost_output,
        "cost_total_usd": cost_total,
        "model": MODEL_NAME,
    }

    return review_text, usage_summary


# ---------------- FEEDBACK CSV (OPTIONAL) ----------------

def save_feedback_csv(
    csv_path: Path,
    tool_name: str,
    run_id: str,
    filename: str,
    rating: str,
    comments: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a single row of feedback to a CSV file."""
    csv_exists = csv_path.exists()
    extra = extra or {}

    fieldnames = [
        "timestamp_utc",
        "tool_name",
        "run_id",
        "filename",
        "rating",
        "comments",
        "extra",
    ]

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
        "extra": extra,
    }

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------- STREAMLIT UI (EMBEDDABLE) ----------------

def main(embed: bool = False) -> None:
    """
    TI AMEP Review UI.

    If embed=True:
        - Do NOT call st.set_page_config
        - Assume outer app (app.py) controls header, logo, tabs, etc.
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – TI AMEP Review",
            layout="wide",
        )

    st.subheader("City of Buckeye – TI AMEP Review")
    st.markdown(
        "Automated Tenant Improvement (TI) AMEP review with adopted-code "
        "context (IBC 2024 and related codes)."
    )

    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_api_key:
        st.error(
            "OPENAI_API_KEY is not set in the environment. "
            "Please export it before running the review."
        )
        st.stop()

    client = get_client(env_api_key)

    # Session state for results
    if "ti_review" not in st.session_state:
        st.session_state["ti_review"] = ""
    if "ti_usage" not in st.session_state:
        st.session_state["ti_usage"] = {}
    if "ti_pdf_bytes" not in st.session_state:
        st.session_state["ti_pdf_bytes"] = None
    if "ti_filename" not in st.session_state:
        st.session_state["ti_filename"] = ""

    col1, col2 = st.columns([1.4, 1])

    with col1:
        project_description = st.text_area(
            "Project Description (optional but recommended)",
            height=120,
            help="Briefly describe the TI scope, occupancy, and any key information.",
        )

        uploaded_file = st.file_uploader(
            "Upload TI Plan Set PDF",
            type=["pdf"],
        )

        run_review = st.button("Run TI AMEP Review", type="primary")

    with col2:
        st.markdown("### Code Context Status")
        if CODE_PDF_DIR.exists() and CODE_PDF_DIR.is_dir():
            pdf_list = [p.name for p in CODE_PDF_DIR.glob("*.pdf")]
            if pdf_list:
                st.success(
                    f"Code references loaded ({len(pdf_list)} IBC 2024 PDF files)."
                )
            else:
                st.warning(
                    "Code folder found, but no IBC 2024 PDF files were detected."
                )
        else:
            st.error(
                "IBC 2024 code folder not found. "
                "Create an 'IBC 2024' folder next to ti_amep_app.py and "
                "place your IBC PDFs inside it."
            )

    if run_review:
        if not uploaded_file:
            st.error("Please upload a TI plan PDF before running the review.")
            st.stop()

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = Path(tmp.name)

            with st.spinner("Running TI AMEP review..."):
                review_text, usage_summary = run_review_pipeline_single(
                    client=client,
                    pdf_path=temp_path,
                    project_description=project_description.strip() or None,
                )

            st.session_state["ti_review"] = review_text
            st.session_state["ti_usage"] = usage_summary
            st.session_state["ti_filename"] = uploaded_file.name

            # Build a PDF version in memory, including run log
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                create_review_pdf(review_text, usage_summary, Path(tmp_pdf.name))
                tmp_pdf.flush()
                with open(tmp_pdf.name, "rb") as f_pdf:
                    st.session_state["ti_pdf_bytes"] = f_pdf.read()

            st.success("TI AMEP review completed.")

        except Exception as e:
            st.error(f"Error during review: {e}")

        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # OUTPUT SECTION
    if st.session_state.get("ti_review"):
        st.markdown("## Review Output (formatted)")
        # Render markdown so headings, bold, and table look nice in the app
        st.markdown(st.session_state["ti_review"])

        st.markdown("### Token Usage & Cost Estimate")
        usage = st.session_state.get("ti_usage", {})
        if usage:
            st.json(usage)

        if st.session_state.get("ti_pdf_bytes"):
            fn = st.session_state.get("ti_filename") or "TI_Plans"
            st.download_button(
                label="Download Review as PDF",
                data=st.session_state["ti_pdf_bytes"],
                file_name=f"buckeye_ti_amep_review_{fn}.pdf",
                mime="application/pdf",
            )

        st.markdown("### Reviewer Feedback (internal)")
        rating = st.radio(
            "How accurate was this review?",
            ["Looks good", "Mostly okay", "Needs corrections"],
            index=0,
            key="ti_rating",
        )
        comments = st.text_area(
            "Notes / corrections (optional)",
            key="ti_comments",
        )

        if st.button("Save Feedback"):
            feedback_dir = BASE_DIR / "feedback_logs"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            csv_path = feedback_dir / "ti_amep_feedback.csv"

            extra = st.session_state.get("ti_usage", {}).copy()
            extra["review_filename"] = st.session_state.get("ti_filename", "")
            run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            save_feedback_csv(
                csv_path=csv_path,
                tool_name="TI_AMEP_REVIEW",
                run_id=run_id,
                filename=st.session_state.get("ti_filename", ""),
                rating=rating,
                comments=comments,
                extra=extra,
            )

            st.success(
                f"Feedback saved to {csv_path.name} in {csv_path.parent}."
            )


# This is what your tabbed app.py can call:
def app():
    """Entry point for multi-tab app: call from app.py."""
    main(embed=True)


if __name__ == "__main__":
    # Standalone run (no outer tabs)
    main(embed=False)
