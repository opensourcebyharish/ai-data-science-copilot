"""AI Data Science Copilot — application entry point.

Milestone 2: this file wires the design system and the navigation
together. The pages themselves live in the views/ folder. Run with:

    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from ui.theme import inject_theme

VIEWS = Path(__file__).resolve().parent / "views"

st.set_page_config(
    page_title="AI Data Science Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

navigation = st.navigation(
    {
        "Workspace": [
            st.Page(VIEWS / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page(VIEWS / "data.py", title="Data", icon=":material/database:"),
            st.Page(VIEWS / "charts.py", title="Charts", icon=":material/monitoring:"),
        ],
        "Intelligence": [
            st.Page(VIEWS / "ai.py", title="AI Assistant", icon=":material/auto_awesome:"),
            st.Page(VIEWS / "models.py", title="Models", icon=":material/model_training:"),
        ],
    },
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <span class="sidebar-brand__mark" aria-hidden="true">📊</span>
          <span class="sidebar-brand__text">
            <span class="sidebar-brand__name">AI Data Science Copilot</span>
            <span class="sidebar-brand__tagline">Analytics workspace</span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.badge(
        "M7 · Compare & Explain",
        color="blue",
        help="Current milestone: multi-model comparison on a shared split with honest best-model selection and feature attribution. Verified milestones are marked done on the Overview page.",
    )
    st.divider()

navigation.run()
