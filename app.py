"""
Layers Analysis – Interactive Dashboard
========================================
Streamlit app for exploring SA and SI geological data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Layers Analysis",
    layout="wide",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load Excel, coerce Sample to float, sort descending."""
    df = pd.read_excel(path)
    df["Sample"] = pd.to_numeric(df["Sample"], errors="coerce")
    df = df.sort_values("Sample", ascending=False).reset_index(drop=True)
    return df


def get_param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Sample"]


def parse_depth_ranges(text: str, min_depth: float, max_depth: float):
    if not text or not text.strip():
        return None
    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split("-")
        nums = []
        i = 0
        while i < len(tokens):
            tok = tokens[i].strip()
            if tok == "" and i + 1 < len(tokens):
                i += 1
                nums.append(-float(tokens[i].strip()))
            else:
                nums.append(float(tok))
            i += 1
        if len(nums) == 1:
            nums = [nums[0], nums[0]]
        if len(nums) != 2:
            return None
        lo, hi = sorted(nums)
        ranges.append((lo, hi))
    ranges.sort(key=lambda r: r[1], reverse=True)
    return ranges if ranges else None


# ─── Outlier / skewness detection ─────────────────────────────────────────────

def detect_skewed_params(
    dataframes: dict[str, pd.DataFrame],
    params: list[str],
    skew_threshold: float = 2.0,
    cv_threshold: float = 3.0,
) -> list[str]:
    """
    Flag parameters that are highly right-skewed and benefit from log-scale.
    A parameter is flagged when EITHER:
      • |skewness| >= skew_threshold  (heavy distributional tail), OR
      • coefficient of variation (std/mean) >= cv_threshold (huge relative spread).
    Computed on pooled non-null positive values across all supplied DataFrames.
    """
    flagged = set()
    for p in params:
        vals = pd.concat(
            [df[p].dropna() for df in dataframes.values() if p in df.columns]
        )
        vals = vals[vals > 0]
        if len(vals) < 5:
            continue
        skew = vals.skew()
        cv = vals.std() / vals.mean() if vals.mean() != 0 else 0
        if abs(skew) >= skew_threshold or cv >= cv_threshold:
            flagged.add(p)
    return sorted(flagged)


def remove_outliers_iqr(
    df: pd.DataFrame, params: list[str], factor: float = 1.5
) -> pd.DataFrame:
    """
    Tukey IQR fence: replace values outside [Q1 − k·IQR, Q3 + k·IQR] with NaN.
    Returns a copy — the original DataFrame is never mutated.
    """
    df = df.copy()
    for p in params:
        if p not in df.columns:
            continue
        s = df[p].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        df.loc[df[p].notna() & ((df[p] < lo) | (df[p] > hi)), p] = np.nan
    return df


# ─── Colors & constants ──────────────────────────────────────────────────────
COLOR_SA = "#1f77b4"
COLOR_SI = "#d62728"
MARKER_SIZE = 5


# ─── Main figure builder ─────────────────────────────────────────────────────

def build_figure(
    dataframes: dict[str, pd.DataFrame],
    params: list[str],
    title: str,
    depth_ranges: list[tuple[float, float]] | None = None,
    height_per_range: int = 600,
    log_params: set[str] | None = None,
):
    if not params:
        return go.Figure()

    if log_params is None:
        log_params = set()

    n_cols = len(params)

    if depth_ranges is None:
        all_samples = pd.concat([d["Sample"] for d in dataframes.values()])
        depth_ranges = [(all_samples.min(), all_samples.max())]

    n_rows = len(depth_ranges)
    fig_height = height_per_range * n_rows

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        vertical_spacing=0.06 if n_rows > 1 else 0.05,
        subplot_titles=params if n_rows == 1 else [
            p if row_idx == 0 else ""
            for row_idx in range(n_rows)
            for p in params
        ],
    )

    palette = [COLOR_SA, COLOR_SI, "#2ca02c", "#9467bd"]
    colors = {key: palette[i] for i, key in enumerate(dataframes.keys())}

    for row_idx, (d_lo, d_hi) in enumerate(depth_ranges):
        row = row_idx + 1
        for col_idx, param in enumerate(params):
            col = col_idx + 1
            for label, df in dataframes.items():
                mask = df["Sample"].between(d_lo, d_hi)
                sub = (
                    df.loc[mask].dropna(subset=[param])
                    if param in df.columns
                    else pd.DataFrame()
                )
                if sub.empty:
                    continue
                show_legend = (
                    row_idx == 0 and col_idx == 0 and len(dataframes) > 1
                )
                fig.add_trace(
                    go.Scatter(
                        x=sub[param],
                        y=sub["Sample"],
                        mode="lines+markers",
                        marker=dict(size=MARKER_SIZE, color=colors.get(label, "#333")),
                        line=dict(width=1.5, color=colors.get(label, "#333")),
                        name=label,
                        legendgroup=label,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{label} – {param}</b><br>"
                            "Depth: %{y} m<br>"
                            "Value: %{x}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        # ── Y-axis: min at bottom, max at top (standard orientation) ──
        # range=[lo, hi] with NO autorange="reversed" → 0 at bottom, max at top.
        fig.update_yaxes(
            range=[d_lo, d_hi],
            autorange=False,
            title_text="Meters" if row == 1 else "",
            row=row,
            col=1,
        )
        for c in range(2, n_cols + 1):
            fig.update_yaxes(
                range=[d_lo, d_hi],
                autorange=False,
                row=row,
                col=c,
            )

    # ── X-axes: vertical tick labels, soft gridlines, optional log ──
    for col_idx, param in enumerate(params):
        for row_idx in range(n_rows):
            fig.update_xaxes(
                tickangle=-90,
                showgrid=True,
                gridcolor="rgba(200,200,200,0.35)",
                gridwidth=1,
                type="log" if param in log_params else "linear",
                row=row_idx + 1,
                col=col_idx + 1,
            )

    fig.update_layout(
        height=fig_height,
        title_text=title,
        title_x=0.5,
        hovermode="closest",
        margin=dict(l=60, r=20, t=80, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
    )

    return fig


# ─── Load original data ──────────────────────────────────────────────────────
df_sa_orig = load_data("data/SA_dados.xlsx")
df_si_orig = load_data("data/SI_dados.xlsx")

if "df_sa" not in st.session_state:
    st.session_state.df_sa = df_sa_orig.copy()
if "df_si" not in st.session_state:
    st.session_state.df_si = df_si_orig.copy()
if "data_modified" not in st.session_state:
    st.session_state.data_modified = False

df_sa: pd.DataFrame = st.session_state.df_sa
df_si: pd.DataFrame = st.session_state.df_si

# ─── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

# 1 · Datasets
st.sidebar.subheader("1 · Datasets to display")
show_sa = st.sidebar.checkbox("SA", value=True)
show_si = st.sidebar.checkbox("SI", value=True)
show_combined = st.sidebar.checkbox("SA + SI (overlay)", value=False)

# 2 · Parameters
st.sidebar.subheader("2 · Parameters")
all_params = sorted(
    set(get_param_cols(df_sa) + get_param_cols(df_si)),
    key=lambda p: (
        get_param_cols(df_si_orig).index(p)
        if p in get_param_cols(df_si_orig)
        else 99
    ),
)
select_all = st.sidebar.checkbox("Select all parameters", value=True)
if select_all:
    selected_params = all_params
else:
    selected_params = st.sidebar.multiselect(
        "Choose parameters",
        options=all_params,
        default=all_params,
    )

# 3 · Depth range
st.sidebar.subheader("3 · Depth range filter")
st.sidebar.caption(
    "Leave empty for full range. Enter comma-separated ranges "
    "(e.g. `70-50, 4-2`). Order doesn't matter."
)
depth_input = st.sidebar.text_input(
    "Depth ranges", value="", placeholder="e.g. 70-50, 10-5"
)
depth_ranges = (
    parse_depth_ranges(
        depth_input,
        min(df_sa["Sample"].min(), df_si["Sample"].min()),
        max(df_sa["Sample"].max(), df_si["Sample"].max()),
    )
    if depth_input.strip()
    else None
)
if depth_input.strip() and depth_ranges is None:
    st.sidebar.error("Could not parse depth ranges. Use format: `70-50, 4-2`")

# 4 · Outlier / log-scale handling
st.sidebar.subheader("4 · Outlier handling")

skewed = detect_skewed_params({"SA": df_sa, "SI": df_si}, selected_params)

use_log = st.sidebar.checkbox("Apply log-scale to skewed parameters", value=False)
if use_log and skewed:
    st.sidebar.caption(f"Auto-detected: **{', '.join(skewed)}**")
elif use_log and not skewed:
    st.sidebar.caption("No parameters detected as highly skewed.")
log_params: set[str] = set(skewed) if use_log else set()

use_outlier_removal = st.sidebar.checkbox("Remove outliers (IQR method)", value=False)
iqr_factor = 1.5
if use_outlier_removal:
    iqr_factor = st.sidebar.slider(
        "IQR fence multiplier (k)",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help=(
            "Values outside [Q1 − k·IQR, Q3 + k·IQR] are treated as outliers. "
            "Lower k → more aggressive. 1.5 is the standard Tukey fence."
        ),
    )

# Prepare plot-ready DataFrames (outliers optionally removed)
if use_outlier_removal:
    df_sa_plot = remove_outliers_iqr(df_sa, selected_params, factor=iqr_factor)
    df_si_plot = remove_outliers_iqr(df_si, selected_params, factor=iqr_factor)
else:
    df_sa_plot = df_sa
    df_si_plot = df_si

# ─── Main area ────────────────────────────────────────────────────────────────
st.title("Layers Analysis")

if st.session_state.data_modified:
    st.warning(
        "⚠️ **Data has been modified.** The graphs and tables below reflect "
        "edited data, NOT the original spreadsheets. Reset in the table section below."
    )

if use_outlier_removal:
    st.info(
        f"🔬 **Outlier removal active** (IQR × {iqr_factor}). "
        "Outlier values are hidden from graphs but remain in the tables."
    )

if not selected_params:
    st.info("Select at least one parameter in the sidebar to display graphs.")
    st.stop()

# ─── Graphs ───────────────────────────────────────────────────────────────────
if show_sa:
    st.plotly_chart(
        build_figure(
            {"SA": df_sa_plot},
            selected_params,
            "SA – Vertical Profiles",
            depth_ranges,
            log_params=log_params,
        ),
        use_container_width=True,
        key="chart_sa",
    )

if show_si:
    st.plotly_chart(
        build_figure(
            {"SI": df_si_plot},
            selected_params,
            "SI – Vertical Profiles",
            depth_ranges,
            log_params=log_params,
        ),
        use_container_width=True,
        key="chart_si",
    )

if show_combined:
    st.plotly_chart(
        build_figure(
            {"SA": df_sa_plot, "SI": df_si_plot},
            selected_params,
            "SA vs SI – Combined Vertical Profiles",
            depth_ranges,
            log_params=log_params,
        ),
        use_container_width=True,
        key="chart_combined",
    )

if not (show_sa or show_si or show_combined):
    st.info("Enable at least one dataset in the sidebar.")

# ─── Data tables ──────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📋 Data Tables")
st.caption(
    "You can edit cells directly. Use the controls below each table to "
    "add rows/columns. Changes are reflected in the graphs above."
)


def render_table_section(
    label: str, key_prefix: str, df: pd.DataFrame, orig: pd.DataFrame
):
    st.subheader(f"{label} Data")

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"{key_prefix}_editor",
    )

    changed = not edited.equals(df)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        new_col_name = st.text_input("New column name", key=f"{key_prefix}_newcol")
    with col2:
        add_col = st.button("Add column", key=f"{key_prefix}_addcol")
    with col3:
        reset = st.button("🔄 Reset to original", key=f"{key_prefix}_reset")

    if add_col and new_col_name.strip():
        if new_col_name.strip() not in edited.columns:
            edited[new_col_name.strip()] = np.nan
            changed = True
        else:
            st.warning(f"Column '{new_col_name.strip()}' already exists.")

    if reset:
        edited = orig.copy()
        changed = False
        st.session_state.data_modified = False
        st.toast(f"{label} data reset to original.")

    if changed:
        st.session_state.data_modified = True

    return edited


df_sa_new = render_table_section("SA", "sa", df_sa, df_sa_orig)
st.session_state.df_sa = df_sa_new

st.markdown("---")

df_si_new = render_table_section("SI", "si", df_si, df_si_orig)
st.session_state.df_si = df_si_new
