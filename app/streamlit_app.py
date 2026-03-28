"""
Embedding Debugger — Streamlit entry point.

Navigation is organized around 3 diagnostic pillars:

  🔴 Retrieval Debugging    — find where retrieval breaks
  🟡 Perturbation & Robustness — stress-test against meaning-altering edits
  🔵 Geometry & Drift       — understand embedding space structure

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Embedding Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Navigation — 3 pillars
# ──────────────────────────────────────────────────────────────────────

PAGES = {
    "🏠 Overview":                   "home",
    # Pillar 1
    "─── Retrieval Debugging ───":    None,
    "🎯  Killer Demo":                "killer_demo",
    "📡  Retrieval Debugger":         "retrieval",
    "🔍  Similarity Inspector":       "similarity",
    # Pillar 2
    "─── Perturbation & Robustness ───": None,
    "⚡  Perturbation Lab":           "perturbation",
    # Pillar 3
    "─── Geometry & Drift ───":       None,
    "🗺️   Cluster Geometry":          "clustering",
    "🤝  Model Comparison":           "comparison",
    "📈  Drift Tracker":              "drift",
    "🚨  Outlier Detector":           "outliers",
}

with st.sidebar:
    st.markdown("## 🔬 Embedding Debugger")
    st.caption("A debugger for embedding failures.")
    st.divider()

    nav_options = [k for k, v in PAGES.items() if v is not None and not k.startswith("─")]
    section_headers = {k for k, v in PAGES.items() if v is None or k.startswith("─")}

    # Render grouped navigation
    choice = None
    for label, page_id in PAGES.items():
        if label.startswith("─"):
            st.markdown(f"<small><b>{label.strip('─ ')}</b></small>", unsafe_allow_html=True)
        elif page_id is not None:
            if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state["current_page"] = page_id

    st.divider()
    st.caption("v0.1.0 · MIT License")

# Persist current page across reruns
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

page = st.session_state["current_page"]

# ──────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────

if page == "home":
    from app.pages import home as pg
elif page == "killer_demo":
    from app.pages import killer_demo as pg
elif page == "retrieval":
    from app.pages import retrieval as pg
elif page == "similarity":
    from app.pages import similarity as pg
elif page == "perturbation":
    from app.pages import perturbation as pg
elif page == "clustering":
    from app.pages import clustering as pg
elif page == "comparison":
    from app.pages import comparison as pg
elif page == "drift":
    from app.pages import drift as pg
elif page == "outliers":
    from app.pages import outliers as pg
else:
    from app.pages import home as pg

pg.render()
