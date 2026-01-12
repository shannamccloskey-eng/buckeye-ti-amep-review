import importlib
import base64
from typing import Callable, Dict, Any, List
from pathlib import Path

import streamlit as st


# -------------------------------------------------------------------
# Registry of tools / sub-apps.
# Each entry: label shown on tab, module name, function name, kwargs
# -------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "label": "Plan Intake",
        "module": "intake_app",   # intake_app.py
        "func": "main",
        "kwargs": {"embed": True},  # intake_app.main(embed=...)
    },
    {
        "label": "TI AMEP Review",
        "module": "ti_amep_app",  # ti_amep_app.py
        "func": "main",
        "kwargs": {"embed": True},  # ti_amep_app.main(embed=...)
    },
    {
        "label": "Geo Summary",
        "module": "geo_app",      # geo_app.py
        "func": "main",
        "kwargs": {"embed": True},  # geo_app.main(embed=...)
    },
    {
        "label": "Residential Review",
        "module": "arch_app",
        "func": "main",
        "kwargs": {"embed": True},
    },
    {
        "label": "Feedback Dashboard",
        "module": "feedback_dashboard_app",  # feedback_dashboard_app.py
        "func": "main",
        "kwargs": {"embed": True},  # feedback_dashboard_app.main(embed=...)
    },
]


def _load_tool_callable(tool: Dict[str, Any]) -> Callable[[], None]:
    """
    Import the module + function specified in TOOLS.
    Returns a zero-arg callable we can run inside the tab.
    If anything fails, returns a function that displays an error in the UI.
    """
    label = tool["label"]
    module_name = tool["module"]
    func_name = tool["func"]
    kwargs = tool.get("kwargs", {})

    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name)

        def _wrapped():
            # Call with kwargs (e.g., embed=True)
            return fn(**kwargs)

        return _wrapped

    except Exception as e:
        msg = (
            f"Tool '{label}' is not available right now "
            f"({module_name}.{func_name}): {e}"
        )

        def _error(message: str = msg):
            st.error(message)

        return _error


def _inject_custom_css() -> None:
    """
    Inject custom CSS for:
    - Buckeye color palette
    - Pill-style colored tabs
    - Card-style tab content
    - Modern buttons, backgrounds, etc.
    - ICC logo button on tab row
    """
    st.markdown(
        """
<style>
:root {
  --buckeye-primary: #c45c26;        /* Buckeye orange */
  --buckeye-primary-soft: #fbe6da;   /* soft orange background */
  --buckeye-dark: #1f2933;
  --buckeye-muted: #6b7280;
  --buckeye-border: #e0e4ea;
  --buckeye-bg: #f5f6fa;
}

/* App background */
[data-testid="stAppViewContainer"] {
  background: var(--buckeye-bg);
}

/* Sidebar (collapsed by default, but keep clean if opened) */
[data-testid="stSidebar"] {
  background-color: white;
  border-right: 1px solid var(--buckeye-border);
}

/* Base typography */
html, body, [data-testid="stAppViewContainer"] {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main container spacing – bumped up to give logo more room */
.block-container {
  padding-top: 1.5rem;
}

/* Logo wrapper – extra padding so it isn't clipped at the top */
.buckeye-logo {
  padding-top: 1.0rem;
  padding-bottom: 0.4rem;
}

.buckeye-logo img {
  display: block;
}

/* ---------------- Tabs styling ---------------- */
div[data-testid="stTabs"] > div[role="tablist"] {
  border-bottom: 1px solid var(--buckeye-border) !important;
  padding-bottom: 0.25rem;
  gap: 0.25rem;
}

/* Tab buttons: pill shape */
div[data-testid="stTabs"] button[role="tab"] {
  border-radius: 999px;
  padding: 0.35rem 1.1rem;
  font-weight: 500;
  color: var(--buckeye-muted);
  border: 1px solid transparent;
  background: transparent;
  box-shadow: none !important;
}

/* Hover */
div[data-testid="stTabs"] button[role="tab"]:hover {
  background: rgba(31,41,51,0.04);
  color: var(--buckeye-dark);
}

/* Active tab */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: white !important;
  background: var(--buckeye-primary) !important;
  border-color: var(--buckeye-primary) !important;
  box-shadow: 0 4px 10px rgba(196,92,38,0.35) !important;
}

/* Remove default underline / highlight from active tab */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::before,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {
  border-bottom: none !important;
  box-shadow: none !important;
}

/* ---------------- Card feel for tab content ---------------- */
div[data-testid="stTabs"] + div {
  margin-top: 0.75rem;
  padding: 1.25rem 1.5rem 1.5rem 1.5rem;
  border-radius: 12px;
  background: white;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
  border: 1px solid var(--buckeye-border);
}

/* ---------------- Buttons (Run / Download / etc.) ---------------- */
.stButton>button,
.stDownloadButton>button {
  border-radius: 999px;
  border: 1px solid transparent;
  background: var(--buckeye-primary);
  color: white;
  font-weight: 600;
  padding: 0.4rem 1.3rem;
  box-shadow: 0 4px 10px rgba(196,92,38,0.25);
  min-width: 260px;
}

.stButton>button:hover,
.stDownloadButton>button:hover {
  background: #ad4f21;
  border-color: #ad4f21;
}

/* Secondary (generic secondary buttons) */
button[kind="secondary"], button[kind="secondary"] * {
  border-radius: 999px !important;
}

/* Info boxes / alerts */
[data-baseweb="notification"] {
  border-radius: 10px;
}

/* Tables */
table {
  border-radius: 8px;
  overflow: hidden;
}

/* Hide default Streamlit footer */
footer, .stApp footer {
  visibility: hidden;
}

/* ---------------- ICC logo button on tab row ---------------- */
.icc-tab-button-wrapper {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;   /* anchor to top of column */
  align-items: center;           /* center label + pill relative to each other */
  margin-top: 0.25rem;
}

.icc-access-label {
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #0b4f6c;
  margin-bottom: 0.25rem;
  text-align: center;            /* center ACCESS over logo */
  width: 100%;
}

.icc-tab-img-link {
  display: inline-block;
}

.icc-tab-img-link img {
  height: 104px;
  width: auto;
  display: block;
  border-radius: 999px;
  background: white;
  padding: 12px 28px;
  box-shadow: 0 8px 24px rgba(11,79,108,0.45);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _image_to_base64(path: Path) -> str:
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""


def main():
    # Global page config – sidebar collapsed so no extra column
    st.set_page_config(
        page_title="City of Buckeye – Plan Review Tools",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_custom_css()

    # ---- Header with Buckeye logo + title ----
    logo_path = Path(__file__).parent / "City of Buckeye 2025.png"

    header_cols = st.columns([1, 3], vertical_alignment="center")
    with header_cols[0]:
        if logo_path.exists():
            st.markdown('<div class="buckeye-logo">', unsafe_allow_html=True)
            st.image(str(logo_path), width=200)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption(
                "Logo file not found – expected 'City of Buckeye 2025.png' next to app.py"
            )

    with header_cols[1]:
        st.title("City of Buckeye – Plan Review Tools")
        st.write(
            "Unified interface for Building Safety tools, including Commercial Plan Intake, "
            "TI AMEP review, Geotechnical summary, Residential review, and the Feedback Dashboard."
        )

    st.markdown("---")

    # ---- Tabs + right-justified ICC button on the same row ----
    tab_col, icc_col = st.columns([7, 1], vertical_alignment="top")

    tab_labels = [tool["label"] for tool in TOOLS]

    with tab_col:
        tabs = st.tabs(tab_labels)

    with icc_col:
        icc_logo_path = Path(__file__).parent / "icc_logo.png"

        if icc_logo_path.exists():
            img_b64 = _image_to_base64(icc_logo_path)
            st.markdown(
                f"""
                <div class="icc-tab-button-wrapper">
                    <div class="icc-access-label">ACCESS</div>
                    <a href="https://codes.iccsafe.org/" target="_blank"
                       class="icc-tab-img-link">
                        <img src="data:image/png;base64,{img_b64}"
                             alt="ICC – International Code Council" />
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Simple text fallback if logo file is missing
            st.markdown(
                """
                <div class="icc-tab-button-wrapper">
                    <div class="icc-access-label">ACCESS</div>
                    <a href="https://codes.iccsafe.org/" target="_blank">
                        ICC Codes
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Render each tool in its tab ----
    for tab, tool in zip(tabs, TOOLS):
        with tab:
            render_tool = _load_tool_callable(tool)
            render_tool()

    # Global footer / caption
    st.caption(
        "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC, "
        "2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and "
        "City of Buckeye Amendments."
    )


if __name__ == "__main__":
    main()
