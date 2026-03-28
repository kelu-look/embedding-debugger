"""
Embedding Debugger — Streamlit entry point.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Embedding Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------

PAGES = {
    "🏠 Home":               "home",
    "🔍 Similarity Inspector": "similarity",
    "⚡ Perturbation Lab":   "perturbation",
    "📡 Retrieval Debugger": "retrieval",
    "🗺️ Cluster Geometry":   "clustering",
    "🤝 Model Comparison":   "comparison",
    "📈 Drift Tracker":      "drift",
    "🚨 Outlier Detector":   "outliers",
}

with st.sidebar:
    st.title("🔬 Embedding Debugger")
    st.caption("Local-first embedding diagnostics")
    st.divider()
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("v0.1.0 · MIT License")

page = PAGES[choice]

# ------------------------------------------------------------------
# Lazy-import and dispatch
# ------------------------------------------------------------------

if page == "home":
    from app.pages import home as pg
elif page == "similarity":
    from app.pages import similarity as pg
elif page == "perturbation":
    from app.pages import perturbation as pg
elif page == "retrieval":
    from app.pages import retrieval as pg
elif page == "clustering":
    from app.pages import clustering as pg
elif page == "comparison":
    from app.pages import comparison as pg
elif page == "drift":
    from app.pages import drift as pg
elif page == "outliers":
    from app.pages import outliers as pg

pg.render()
