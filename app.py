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
    # Future tools – just uncomment/add these when ready:
    # {
    #     "label": "Architectural Review",
    #     "module": "arch_app",
    #     "func": "main",
    #     "kwargs": {"embed": True},
    # },
    # {
    #     "label": "Geotechnical Review",
    #     "module": "geo_app",
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


def main():
    # Global page config – must be called exactly once
    st.set_page_config(
        page_title="City of Buckeye – Plan Review Tools",
        layout="wide",
    )

    # ---- Header with Buckeye logo + title ----
    logo_path = Path(__file__).parent / "City of Buckeye 2025.png"  # <-- change name here if needed

    header_cols = st.columns([1, 3])
    with header_cols[0]:
        if logo_path.exists():
            # width replaces deprecated use_column_width
            st.image(str(logo_path), width=220)
        else:
            st.caption("Logo file not found – expected 'City of Buckeye 2025.png' next to app.py")

    with header_cols[1]:
        st.title("City of Buckeye – Plan Review Tools")
        st.write(
            "Unified interface for Building Safety tools, including Commercial Plan Intake "
            "and TI AMEP review. Additional review tools can be added as new tabs."
        )

    st.markdown("---")

    # Build tabs from TOOLS registry
    tab_labels = [tool["label"] for tool in TOOLS]
    tabs = st.tabs(tab_labels)

    for tab, tool in zip(tabs, TOOLS):
        with tab:
            render_tool = _load_tool_callable(tool)
            render_tool()

    # Global footer
    st.caption(
        "Review performed in accordance with the 2024 IBC, IMC, IPC, IFC, "
        "2017 ICC A117.1, 2023 NEC, 2018 IECC, ADA Standards, and "
        "City of Buckeye Amendments."
    )


if __name__ == "__main__":
    main()
