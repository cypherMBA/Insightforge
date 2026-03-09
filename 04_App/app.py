"""
Step 7c: Streamlit UI — InsightForge
Business Intelligence Assistant with chat interface and interactive visualizations.

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure imports resolve from this directory
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from step1_data_preparation import load_data, compute_summary_statistics
from step7b_visualizations import build_all_figures

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightForge — BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources (loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading data and AI assistant...")
def get_assistant():
    from step6_assistant import InsightForgeAssistant
    return InsightForgeAssistant()


@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_data()


@st.cache_data(show_spinner="Computing statistics...")
def get_stats(_df):
    return compute_summary_statistics(_df)


@st.cache_data(show_spinner="Building visualizations...")
def get_figures(_df):
    return build_all_figures(_df)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant_ready" not in st.session_state:
    st.session_state.assistant_ready = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=80)
    st.title("InsightForge")
    st.caption("AI-Powered Business Intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Chat Assistant", "📈 Visualizations", "📋 Data Summary", "🧪 Evaluation"],
        label_visibility="collapsed",
    )

    st.divider()

    if page == "💬 Chat Assistant":
        st.subheader("Chat Options")
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.assistant_ready:
                get_assistant().reset()
            st.rerun()

        st.divider()
        st.subheader("Suggested Questions")
        suggestions = [
            "What are total sales by region?",
            "Which product performed best?",
            "How did sales trend year over year?",
            "What is the average customer satisfaction?",
            "Which age group spends the most?",
            "Compare Q1 and Q3 sales.",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, use_container_width=True, key=f"sugg_{suggestion}"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

df = get_data()
stats = get_stats(df)
figs = get_figures(df)

# ── PAGE: Chat Assistant ──────────────────────────────────────────────────

if page == "💬 Chat Assistant":
    st.title("💬 InsightForge Chat")
    st.caption("Ask questions about your sales data in plain English.")

    # Load assistant (triggers spinner once)
    assistant = get_assistant()
    st.session_state.assistant_ready = True

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask about sales, products, regions, customers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = assistant.ask(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Handle suggestion clicks (no input box needed)
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user_msg = st.session_state.messages[-1]["content"]
        # Check if this message hasn't been answered yet
        if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = assistant.ask(last_user_msg)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


# ── PAGE: Visualizations ─────────────────────────────────────────────────

elif page == "📈 Visualizations":
    st.title("📈 Sales Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Sales Trends", "Product Analysis", "Regional Analysis", "Demographics & Satisfaction"]
    )

    with tab1:
        st.subheader("Monthly Sales Trend by Year")
        st.plotly_chart(figs["sales_trend"], use_container_width=True)
        st.subheader("Quarterly Sales by Year")
        st.plotly_chart(figs["quarterly_sales"], use_container_width=True)

    with tab2:
        st.subheader("Product Performance Overview")
        st.plotly_chart(figs["product_performance"], use_container_width=True)
        st.subheader("Sales by Product × Region")
        st.plotly_chart(figs["product_region_heatmap"], use_container_width=True)

    with tab3:
        st.subheader("Total Sales by Region")
        st.plotly_chart(figs["regional_sales"], use_container_width=True)
        st.subheader("Monthly Sales Trend by Region")
        st.plotly_chart(figs["regional_trend"], use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gender Distribution")
            st.plotly_chart(figs["gender_distribution"], use_container_width=True)
        with col2:
            st.subheader("Sales by Age Group")
            st.plotly_chart(figs["age_group_sales"], use_container_width=True)

        st.subheader("Sales by Product and Gender")
        st.plotly_chart(figs["gender_sales_by_product"], use_container_width=True)
        st.subheader("Customer Satisfaction: Product × Region")
        st.plotly_chart(figs["satisfaction_heatmap"], use_container_width=True)


# ── PAGE: Data Summary ───────────────────────────────────────────────────

elif page == "📋 Data Summary":
    st.title("📋 Data Summary")

    o = stats["overall"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${o['total_sales']:,}")
    col2.metric("Avg Daily Sale", f"${o['mean_sales']:,}")
    col3.metric("Median Sale", f"${o['median_sales']:,}")
    col4.metric("Std Deviation", f"${o['std_sales']:,}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Sales by Product")
        import pandas as pd
        product_df = pd.DataFrame(stats["by_product"]).T.reset_index()
        product_df.columns = ["Product", "Total", "Mean", "Median", "Count"]
        product_df["Total"] = product_df["Total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(product_df, use_container_width=True, hide_index=True)

        st.subheader("Sales by Region")
        region_df = pd.DataFrame(stats["by_region"]).T.reset_index()
        region_df.columns = ["Region", "Total", "Mean", "Median", "Count"]
        region_df["Total"] = region_df["Total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(region_df, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("Annual Sales")
        year_df = pd.DataFrame(stats["by_year"]).T.reset_index()
        year_df.columns = ["Year", "Total", "Mean", "Median"]
        year_df["Total"] = year_df["Total"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(year_df, use_container_width=True, hide_index=True)

        st.subheader("Customer Satisfaction")
        s = stats["satisfaction"]
        st.metric("Overall Mean", f"{s['overall_mean']} / 5.0")
        sat_df = pd.DataFrame({
            "Product": list(s["by_product"].keys()),
            "Avg Satisfaction": list(s["by_product"].values()),
        })
        st.dataframe(sat_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)


# ── PAGE: Evaluation ────────────────────────────────────────────────────

elif page == "🧪 Evaluation":
    st.title("🧪 Model Evaluation")
    st.caption("Assess InsightForge's accuracy using QAEvalChain against ground-truth Q&A pairs.")

    report_path = os.path.join(
        os.path.dirname(__file__), "..", "05_Evaluation", "eval_report.json"
    )

    if os.path.exists(report_path):
        import json
        with open(report_path) as f:
            results = json.load(f)

        correct = sum(1 for r in results if "CORRECT" in r["grade"].upper())
        total = len(results)
        accuracy = correct / total * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.1f}%")
        col2.metric("Correct", f"{correct}/{total}")
        col3.metric("Model", "gpt-4o-mini")

        st.divider()
        for r in results:
            status = "✅" if "CORRECT" in r["grade"].upper() else "❌"
            with st.expander(f"{status} Q{r['index']}: {r['query']}"):
                col_a, col_b = st.columns(2)
                st.markdown("**Expected Answer**")
                st.info(r["expected"])
    else:
        st.warning("No evaluation report found. Run `step7a_evaluation.py` first to generate results.")
        if st.button("▶ Run Evaluation Now", type="primary"):
            with st.spinner("Running evaluation — this may take a minute..."):
                from step7a_evaluation import run_evaluation, save_report
                assistant = get_assistant()
                results = run_evaluation(assistant)
                save_report(results, report_path)
            st.success("Evaluation complete!")
            st.rerun()
