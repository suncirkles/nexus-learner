"""
app.py
------
Streamlit entry point for Nexus Learner.
Handles page config, global CSS, sidebar navigation, and dispatches to
ui/pages/* view modules. Shrunk from 1,561 lines to ~80 lines as part of
Phase 1 refactoring (repository layer + app.py decomposition).
"""

import streamlit as st
import logging
import uuid

# Logging must be configured before any other local imports so every module
# that calls logging.getLogger(__name__) at import time inherits the setup.
from core.logging_config import setup_logging
setup_logging()

# Configure Page (must be first Streamlit call)
st.set_page_config(page_title="Nexus Learner - Agentic Learning Platform", layout="wide", page_icon="🎓")

logger = logging.getLogger(__name__)

# Global CSS — shared across all pages
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stCard {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 15px;
    }
    .topic-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .topic-header { color: #58a6ff; font-size: 1.6rem; font-weight: 700; margin-bottom: 5px; }
    .subtopic-header { color: #8b949e; font-size: 1.1rem; font-weight: 500; }
    .critic-score {
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        background-color: #238636;
        color: white;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); }
    .subject-tile {
        background: linear-gradient(135deg, #232731 0%, #1a1c24 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .subject-tile:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border-color: #58a6ff;
    }
    .subject-title {
        color: #58a6ff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .stat-label { color: #8b949e; }
    .stat-value { color: #c9d1d9; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("🎓 Nexus Learner")

    nav_options = [
        "🏠 Dashboard",
        "📂 Knowledge Library",
        "📚 Study Materials",
        "👨‍🏫 Mentor Review",
        "🧠 Learner",
        "⚙️ System Tools",
    ]

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "sidebar_nav" not in st.session_state:
        st.session_state.sidebar_nav = nav_options[0]

    # Consume any pending programmatic navigation before the radio is rendered
    if "pending_nav" in st.session_state:
        st.session_state.sidebar_nav = st.session_state.pop("pending_nav")

    # Radio is the single source of truth for navigation
    active_nav = st.sidebar.radio("Navigation", nav_options, key="sidebar_nav")
    st.session_state.active_nav = active_nav

    with st.sidebar:
        from ui.components.background_monitor import render_sidebar_background_monitor
        render_sidebar_background_monitor()

    # Lazy page imports — only load the module the user is actually visiting
    if active_nav == "🏠 Dashboard":
        from ui.pages.dashboard import render_dashboard
        render_dashboard()
    elif active_nav == "📂 Knowledge Library":
        from ui.pages.library import render_knowledge_library
        render_knowledge_library()
    elif active_nav == "📚 Study Materials":
        from ui.pages.study_materials import render_study_materials
        render_study_materials()
    elif active_nav == "👨‍🏫 Mentor Review":
        from ui.pages.mentor import render_mentor_review
        render_mentor_review()
    elif active_nav == "🧠 Learner":
        from ui.pages.learner import render_learner_view
        render_learner_view()
    elif active_nav == "⚙️ System Tools":
        from ui.pages.system_tools import render_system_tools
        render_system_tools()


if __name__ == "__main__":
    main()
