"""
Clinical Trial Inspector Agent Application.

This is the main Streamlit application script. It orchestrates:
1.  **LLM & Agents**: Initializes Google Gemini and the LangChain agent.
2.  **RAG Pipeline**: Loads the LlamaIndex vector store for semantic retrieval.
3.  **User Interface**: Renders the Streamlit UI with tabs for Chat, Analytics, and Raw Data.
4.  **Visualization**: Handles dynamic chart generation using Altair.
"""
import streamlit as st
import pandas as pd
import os
import altair as alt
import logging
from dotenv import load_dotenv

# Suppress logging
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Module Imports
from modules.utils import load_index
from modules.tools import search_trials, find_similar_studies, get_study_analytics

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

# --- App Configuration ---
st.set_page_config(page_title="Clinical Trial Inspector", layout="wide")

st.title("🧬 Clinical Trial Inspector Agent")

# 1. Setup LLM
# We use Google Gemini-2.5-Flash for fast and accurate responses.
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set GOOGLE_API_KEY in .env")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Load LlamaIndex (Cached)
# The index is loaded once and cached to avoid reloading on every interaction.
index = load_index()

# 3. Define Agent
# The agent has access to specific tools for searching and analyzing data.
tools = [search_trials, find_similar_studies, get_study_analytics]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Clinical Trial Expert Assistant. "
            "Your goal is to help researchers and analysts understand clinical trial data. "
            "You have access to a local database of clinical trials (embedded from ClinicalTrials.gov). "
            "Use the available tools to search for studies, find similar studies, and generate analytics. "
            "When asked about 'trends', 'counts', or 'most common', ALWAYS use the `get_study_analytics` tool. "
            "When asked to 'find studies' or 'search', use `search_trials`. "
            "If the user asks for a specific study by ID (e.g., NCT12345678), `search_trials` handles that automatically. "
            "Provide concise, evidence-based answers citing specific studies when possible.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. UI Layout: Tabs for Chat vs Analytics
tab_chat, tab_analytics, tab_raw_data = st.tabs(
    ["💬 Chat Assistant", "📊 Analytics Dashboard", "📂 Raw Data"]
)

# --- TAB 1: CHAT ---
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Render chart if present in message history (persisted charts)
            if "chart_data" in message:
                chart_data = message["chart_data"]
                st.caption(chart_data["title"])
                chart = (
                    alt.Chart(pd.DataFrame(chart_data["data"]))
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            chart_data["x"], sort="-y", axis=alt.Axis(labelLimit=200)
                        ),
                        y=alt.Y(chart_data["y"], title="Count"),
                        tooltip=[chart_data["x"], chart_data["y"]],
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

    # Chat Input
    if prompt := st.chat_input("Ask about clinical trials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing clinical trials..."):
                try:
                    # Clear previous inline chart data to avoid stale charts
                    if "inline_chart_data" in st.session_state:
                        del st.session_state["inline_chart_data"]

                    # Construct chat history for the agent context
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    # Invoke Agent
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": chat_history}
                    )
                    output = response["output"]
                    st.markdown(output)

                    # Check for inline chart data (set by tools)
                    chart_data = None
                    if "inline_chart_data" in st.session_state:
                        chart_data = st.session_state["inline_chart_data"]
                        st.caption(chart_data["title"])
                        if chart_data["type"] == "bar":
                            # Use Altair for better charts
                            chart = (
                                alt.Chart(pd.DataFrame(chart_data["data"]))
                                .mark_bar()
                                .encode(
                                    x=alt.X(
                                        chart_data["x"],
                                        sort="-y",
                                        axis=alt.Axis(labelLimit=200),
                                    ),
                                    y=alt.Y(chart_data["y"], title="Count"),
                                    tooltip=[chart_data["x"], chart_data["y"]],
                                )
                                .interactive()
                            )
                            st.altair_chart(chart, use_container_width=True)

                        # Clean up session state
                        del st.session_state["inline_chart_data"]

                    # Save message with chart data if present
                    msg_obj = {"role": "assistant", "content": output}
                    if chart_data:
                        msg_obj["chart_data"] = chart_data
                    st.session_state.messages.append(msg_obj)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab_analytics:
    st.header("📊 Global Analytics")
    st.write(
        "Analyze trends across the entire clinical trial dataset (60,000+ studies)."
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Configuration")
        group_by = st.selectbox(
            "Group By", ["Phase", "Status", "Sponsor", "Start Year"], index=2
        )

        # Optional Filters
        st.markdown("---")
        st.markdown("**Filters (Optional)**")
        filter_phase = st.text_input("Phase (e.g., Phase 2)", key="dash_phase")
        filter_sponsor = st.text_input("Sponsor (e.g., Pfizer)", key="dash_sponsor")

        run_analytics = st.button("Generate Analytics", type="primary")

    with col2:
        if run_analytics:
            with st.spinner(f"Analyzing studies by {group_by}..."):
                # Map UI selection to tool arguments
                group_map = {
                    "Phase": "phase",
                    "Status": "status",
                    "Sponsor": "sponsor",
                    "Start Year": "start_year",
                }

                # Call the tool directly (bypassing the agent for direct analytics)
                result = get_study_analytics(
                    query="overall",  # Dummy query for "all"
                    group_by=group_map[group_by],
                    phase=filter_phase if filter_phase else None,
                    sponsor=filter_sponsor if filter_sponsor else None,
                )

                # The tool sets session state 'inline_chart_data'
                if "inline_chart_data" in st.session_state:
                    c_data = st.session_state["inline_chart_data"]
                    st.subheader(c_data["title"])

                    # Altair Chart Rendering
                    if group_by == "Start Year":
                        # Line chart for years
                        chart = (
                            alt.Chart(pd.DataFrame(c_data["data"]))
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(
                                    c_data["x"], axis=alt.Axis(format="d"), title="Year"
                                ),  # 'd' for integer year
                                y=alt.Y(c_data["y"], title="Count"),
                                tooltip=[c_data["x"], c_data["y"]],
                            )
                            .interactive()
                        )
                    else:
                        # Bar chart for others
                        chart = (
                            alt.Chart(pd.DataFrame(c_data["data"]))
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    c_data["x"],
                                    sort="-y",
                                    axis=alt.Axis(labelLimit=200),
                                ),
                                y=alt.Y(c_data["y"], title="Count"),
                                tooltip=[c_data["x"], c_data["y"]],
                            )
                            .interactive()
                        )

                    st.altair_chart(chart, use_container_width=True)

                    # Show raw table
                    with st.expander("View Source Data"):
                        st.dataframe(pd.DataFrame(c_data["data"]))

                    # Clean up
                    del st.session_state["inline_chart_data"]
                else:
                    st.warning(result)

# --- TAB 3: RAW DATA ---
with tab_raw_data:
    st.header("📂 Raw Data Explorer")
    st.write("View and filter the underlying dataset.")

    # Load a sample or full dataset? Full might be slow.
    # We load a sample (top 100) to avoid performance issues.
    if st.button("Load Sample Data (Top 100)"):
        with st.spinner("Fetching data..."):
            retriever = index.as_retriever(similarity_top_k=100)
            nodes = retriever.retrieve("clinical trial")
            data = [n.metadata for n in nodes]
            df_raw = pd.DataFrame(data)

            # Format Year to remove commas (e.g., 2,023 -> 2023)
            if "start_year" in df_raw.columns:
                df_raw["start_year"] = (
                    pd.to_numeric(df_raw["start_year"], errors="coerce")
                    .astype("Int64")
                    .astype(str)
                    .str.replace(",", "")
                )

            st.dataframe(
                df_raw,
                column_config={
                    "nct_id": "NCT ID",
                    "title": "Study Title",
                    "start_year": st.column_config.TextColumn(
                        "Start Year"
                    ),  # Force text to avoid commas
                    "url": st.column_config.LinkColumn("URL"),
                },
                use_container_width=True,
                hide_index=True,
            )
