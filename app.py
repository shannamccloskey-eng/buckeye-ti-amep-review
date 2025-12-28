import importlib
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
        "label": "Geotechnical Review",
        "module": "geo_app",      # geo_app.py
        "func": "main",
        "kwargs": {"embed": True},  # geo_app.main(embed=...)
    },
    # Future tools – e.g.:
    # {
    #     "label": "Architectural Review",
    #     "module": "arch_app",
    #     "func": "main",
    #     "kwargs": {"embed": True},
    # },
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
        def _error():
            st.error(
                f"Error loading tool '{label}' "
                f"({module_name}.{func_name}): {e}"
            )
        return _error


def _inject_custom_css() -> None:
    """
    Inject custom CSS for:
    - Buckeye color palette
    - Pill-style colored tabs
    - Card-style tab content
    - Modern buttons, backgrounds, etc.
    - Extra padding + centering help for logo
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

/* Main container spacing */
.block-container {
  padding-top: 0.75rem;  /* slightly reduced so logo + title sit nicely */
}

/* Logo wrapper – extra padding so it isn't clipped */
.buckeye-logo {
  padding-top: 0.6rem;
  padding-bottom: 0.4rem;
}

/* Make sure the logo image itself never hugs the top edge */
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

/* Remove any default underline / highlight from active tab */
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

/* ---------------- Buttons ---------------- */
.stButton>button {
  border-radius: 999px;
  border: 1px solid transparent;
  background: var(--buckeye-primary);
  color: white;
  font-weight: 600;
  padding: 0.4rem 1.3rem;
  box-shadow: 0 4px 10px rgba(196,92,38,0.25);
}

.stButton>button:hover {
  background: #ad4f21;
  border-color: #ad4f21;
}

/* Secondary (e.g., download) buttons */
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
</style>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Global page config – sidebar collapsed so no extra column
    st.set_page_config(
        page_title="City of Buckeye – Plan Review Tools",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_custom_css()

    # ---- Header with Buckeye logo + title ----
    logo_path = Path(__file__).parent / "City of Buckeye 2025.png"  # change name here if needed

    # Align logo + title vertically in the header row
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
            "TI AMEP review, and Geotechnical review. Additional tools can be added as new tabs."
        )

    st.markdown("---")

    # Build tabs from TOOLS registry
    tab_labels = [tool["label"] for tool in TOOLS]
    tabs = st.tabs(tab_labels)

    for tab
