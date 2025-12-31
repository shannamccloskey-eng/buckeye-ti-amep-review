import csv
import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


FEEDBACK_FILES = [
    ("INTAKE", "feedback_intake.csv"),
    ("TI_AMEP", "feedback_ti_amep.csv"),
    ("GEO_SUMMARY", "feedback_geotech_summary.csv"),
]


def _load_feedback() -> pd.DataFrame:
    """
    Load all known feedback CSVs and return a combined DataFrame.
    Missing files are skipped gracefully.
    """
    frames: List[pd.DataFrame] = []

    for tool_label, filename in FEEDBACK_FILES:
        path = Path(filename)
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # Ensure expected columns exist (older files may be missing some)
        for col in ["tool_name", "filename", "rating", "comments", "timestamp_utc"]:
            if col not in df.columns:
                df[col] = ""

        # If tool_name column is empty, fill it with tool_label for clarity
        df["tool_name"] = df["tool_name"].replace("", tool_label)
        df["tool_name"] = df["tool_name"].fillna(tool_label)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Normalize timestamp as datetime and add a simple date column
    if "timestamp_utc" in combined.columns:
        combined["timestamp_utc"] = pd.to_datetime(
            combined["timestamp_utc"], errors="coerce"
        )
        combined["date"] = combined["timestamp_utc"].dt.date

    # Fill rating/comments with empty string if missing
    combined["rating"] = combined.get("rating", "").fillna("")
    combined["comments"] = combined.get("comments", "").fillna("")

    # Standardize rating labels a bit
    combined["rating"] = combined["rating"].replace(
        {
            "looks good": "Looks good",
            "mostly okay": "Mostly okay",
            "needs corrections": "Needs corrections",
        }
    )

    return combined


def main(embed: bool = False):
    """
    Feedback dashboard UI.

    embed = False → standalone
    embed = True  → called from master tabbed app
    """
    if not embed:
        st.set_page_config(
            page_title="City of Buckeye – Feedback Dashboard",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    st.title("City of Buckeye – Feedback Dashboard")
    st.write(
        "This dashboard aggregates feedback from the Intake, TI AMEP, and "
        "Geotechnical tools. Use it to monitor accuracy, identify recurring "
        "issues, and guide prompt / workflow improvements."
    )

    df = _load_feedback()

    if df.empty:
        st.warning(
            "No feedback data found yet. Once you start saving feedback from the "
            "tools, this dashboard will display summary statistics here.\n\n"
            "Expected files:\n"
            "- feedback_intake.csv\n"
            "- feedback_ti_amep.csv\n"
            "- feedback_geotech_summary.csv"
        )
        return

    # Sidebar / top filters
    st.subheader("Filters")

    tools = sorted(df["tool_name"].dropna().unique().tolist())
    selected_tools = st.multiselect(
        "Filter by tool",
        tools,
        default=tools,
    )

    ratings = sorted(df["rating"].dropna().unique().tolist())
    selected_ratings = st.multiselect(
        "Filter by rating",
        ratings,
        default=ratings,
    )

    # Optional: date range filter
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.date_input(
        "Filter by date range (UTC)",
        value=(min_date, max_date),
    )

    df_filtered = df.copy()

    if selected_tools:
        df_filtered = df_filtered[df_filtered["tool_name"].isin(selected_tools)]

    if selected_ratings:
        df_filtered = df_filtered[df_filtered["rating"].isin(selected_ratings)]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered["date"] >= start_date) & (df_filtered["date"] <= end_date)
        ]

    if df_filtered.empty:
        st.info(
            "No feedback entries match the current filters. Try broadening the "
            "tool, rating, or date selections."
        )
        return

    # --- Summary metrics ---
    st.subheader("Summary Metrics")

    total_entries = len(df_filtered)
    rating_counts = df_filtered["rating"].value_counts().sort_index()
    tool_counts = df_filtered["tool_name"].value_counts().sort_index()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total feedback entries", total_entries)

    if "Looks good" in rating_counts:
        pct_good = 100.0 * rating_counts["Looks good"] / total_entries
        col2.metric("Looks good (%)", f"{pct_good:.1f}%")
    else:
        col2.metric("Looks good (%)", "–")

    if "Needs corrections" in rating_counts:
        pct_needs = 100.0 * rating_counts["Needs corrections"] / total_entries
        col3.metric("Needs corrections (%)", f"{pct_needs:.1f}%")
    else:
        col3.metric("Needs corrections (%)", "–")

    # --- Rating distribution chart ---
    st.subheader("Rating distribution")
    st.bar_chart(rating_counts)

    # --- Feedback by tool chart ---
    st.subheader("Feedback entries by tool")
    st.bar_chart(tool_counts)

    # --- Recent feedback table ---
    st.subheader("Recent feedback entries")

    df_display = df_filtered.sort_values(
        by="timestamp_utc", ascending=False
    ).copy()

    # Keep only the most relevant columns for display
    display_cols = [
        "timestamp_utc",
        "tool_name",
        "filename",
        "rating",
        "comments",
    ]
    extra_cols = [c for c in df_display.columns if c not in display_cols]
    df_display = df_display[display_cols + extra_cols]

    st.dataframe(
        df_display.head(50),
        use_container_width=True,
    )

    # Download filtered data
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered feedback as CSV",
        data=csv_bytes,
        file_name="buckeye_feedback_filtered.csv",
        mime="text/csv",
    )

    st.caption(
        "Feedback data is stored in CSV files next to the app: "
        "`feedback_intake.csv`, `feedback_ti_amep.csv`, and "
        "`feedback_geotech_summary.csv`. These can also be analyzed directly "
        "in Excel, Power BI, or other tools."
    )


if __name__ == "__main__":
    main()
