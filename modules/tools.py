"""
LangChain Tools for the Clinical Trial Agent.

This module defines the tools that the agent can use to interact with the clinical trial data.
Tools include:
1.  **search_trials**: Semantic search with optional strict filtering.
2.  **find_similar_studies**: Finding studies semantically similar to a given text.
3.  **get_study_analytics**: Aggregating data for trends and insights (with inline charts).
"""

import pandas as pd
import streamlit as st
from typing import Optional
from langchain.tools import tool as langchain_tool
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from modules.utils import (
    load_index,
    normalize_sponsor,
    LocalMetadataPostFilter,
    KeywordBoostingPostProcessor,
)
import re

# --- Tools ---


def expand_query(query: str) -> str:
    """Expands a search query with synonyms using the LLM."""
    if not query or len(query.split()) > 10:  # Skip expansion for long queries
        return query

    prompt = (
        f"You are a helpful medical assistant. "
        f"Expand the following search query with relevant medical synonyms and acronyms. "
        f"Return ONLY the expanded query string combined with OR operators. "
        f"Do not add any explanation.\n\n"
        f"Query: {query}\n"
        f"Expanded Query:"
    )
    try:
        # Use the global Settings.llm
        if not Settings.llm:
            # Fallback if not initialized (though load_index does it)
            from modules.utils import setup_llama_index

            setup_llama_index()

        response = Settings.llm.complete(prompt)
        expanded = response.text.strip()
        # Clean up if LLM is chatty
        if "Expanded Query:" in expanded:
            expanded = expanded.split("Expanded Query:")[-1].strip()
        print(f"✨ Expanded Query: '{query}' -> '{expanded}'")
        return expanded
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return query


@langchain_tool("search_trials")
def search_trials(
    query: str = None,
    status: str = None,
    phase: str = None,
    sponsor: str = None,
    intervention: str = None,
    year: int = None,
):
    """
    Searches for clinical trials using semantic search with optional strict filters.

    This tool combines vector-based semantic search with metadata filtering to find
    relevant studies. It supports both pre-retrieval filtering (efficient) and
    post-retrieval filtering (flexible).

    Args:
        query (str, optional): The natural language search query (e.g., "diabetes treatment").
                               If not provided, one is constructed from the filters.
        status (str, optional): Filter by recruitment status (e.g., "RECRUITING", "COMPLETED").
        phase (str, optional): Filter by trial phase (e.g., "PHASE2", "PHASE3").
                               Accepts comma-separated values for multiple phases.
        sponsor (str, optional): Filter by sponsor name (e.g., "Pfizer").
                                 Accepts comma-separated values.
        intervention (str, optional): Filter by intervention/drug name (e.g., "Keytruda").
        year (int, optional): Filter for studies starting on or after this year (e.g., 2020).

    Returns:
        str: A string representation of the search results (top relevant studies).
    """
    index = load_index()

    # --- Query Construction ---
    # Handle missing query by constructing one from filters to ensure vector search has content
    if not query:
        parts = []
        if sponsor:
            parts.append(sponsor)
        if intervention:
            parts.append(intervention)
        if phase:
            parts.append(phase)
        if status:
            parts.append(status)
        query = " ".join(parts) if parts else "clinical trial"
    else:
        # Inject sponsor and phase into query if they are not already present
        # This helps the vector search align with the metadata filters
        if sponsor:
            norm_sponsor = normalize_sponsor(sponsor)
            if norm_sponsor and norm_sponsor.lower() not in query.lower():
                query = f"{norm_sponsor} {query}"
        if intervention and intervention.lower() not in query.lower():
            query = f"{intervention} {query}"
        if phase and phase.lower() not in query.lower():
            query = f"{phase} {query}"

        # Expand query with synonyms
        query = expand_query(query)

    # --- Pre-Retrieval Filters (ChromaDB) ---
    # These filters are applied *before* vector search, reducing the search space.
    filters = []

    # Detect if query is an NCT ID (e.g., NCT01234567)
    # If found, force an exact match on the ID
    nct_match = re.search(r"NCT\d+", query, re.IGNORECASE)
    if nct_match:
        nct_id = nct_match.group(0).upper()
        print(f"🎯 Detected NCT ID: {nct_id}. Switching to exact match.")
        filters.append(
            MetadataFilter(key="nct_id", value=nct_id, operator=FilterOperator.EQ)
        )

    if status:
        filters.append(
            MetadataFilter(
                key="status", value=status.upper(), operator=FilterOperator.EQ
            )
        )
    if year:
        filters.append(
            MetadataFilter(key="start_year", value=year, operator=FilterOperator.GTE)
        )

    metadata_filters = MetadataFilters(filters=filters) if filters else None

    print(
        f"🔍 Tool Called: search_trials(query='{query}', status='{status}', phase='{phase}', sponsor='{sponsor}', intervention='{intervention}')"
    )

    # --- Post-Retrieval Filters (Custom) ---
    # These filters are applied *after* fetching candidates.
    # Useful for complex logic like fuzzy matching or multi-value fields that Chroma might not handle perfectly.
    post_filters = []
    if phase or sponsor or intervention:
        post_filters.append(
            LocalMetadataPostFilter(
                phase=phase, sponsor=sponsor, intervention=intervention
            )
        )

    # --- Hybrid Search Tuning ---
    # Boost results with exact keyword matches in Title/ID
    post_filters.append(KeywordBoostingPostProcessor())

    # --- Re-Ranking ---
    # Use a Cross-Encoder to re-score the top results for better relevance.
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=5
    )
    post_filters.append(reranker)

    # Increase retrieval limit if filters are present to ensure we get enough candidates
    # before post-filtering reduces the count.
    top_k = 200 if (phase or sponsor or intervention) else 50

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=post_filters,
        filters=metadata_filters,
    )
    response = query_engine.query(query)

    # --- Self-Correction / Retry Logic ---
    if not response.source_nodes and (phase or sponsor or intervention):
        print(
            "⚠️ No results found with strict filters. Retrying with relaxed filters..."
        )
        # Retry without the strict LocalMetadataPostFilter, but keep the Re-Ranker
        relaxed_post_filters = [reranker]

        query_engine_relaxed = index.as_query_engine(
            similarity_top_k=50,
            node_postprocessors=relaxed_post_filters,
            filters=metadata_filters,  # Keep pre-retrieval filters (Status, Year) as they are usually hard constraints
        )
        response = query_engine_relaxed.query(query)
        if response.source_nodes:
            return f"No exact matches found for your strict filters. Here are some semantically relevant studies for '{query}':\n{response}"

    print(f"✅ Retrieved {len(response.source_nodes)} nodes after filtering.")
    return str(response)


@langchain_tool("find_similar_studies")
def find_similar_studies(query: str):
    """
    Finds studies semantically similar to a given query or study description.

    This tool is useful for "more like this" functionality. It relies purely
    on vector similarity without strict metadata filtering.

    Args:
        query (str): The text to match against (e.g., a study title or description).

    Returns:
        str: A string containing the top 5 similar studies with their titles and summaries.
    """
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        results.append(
            f"Study: {node.metadata['title']} (Score: {node.score:.2f})\nSummary: {node.text[:200]}..."
        )

    return "\n\n".join(results)


@langchain_tool("get_study_analytics")
def get_study_analytics(
    query: str,
    group_by: str,
    phase: Optional[str] = None,
    status: Optional[str] = None,
    sponsor: Optional[str] = None,
    intervention: Optional[str] = None,
):
    """
    Aggregates clinical trial data based on a search query and groups by a specific field.

    This tool performs the following steps:
    1.  Retrieves a large number of relevant studies (up to 500).
    2.  Applies strict filters (Phase, Status, Sponsor) in memory (Pandas).
    3.  Groups the data by the requested field (e.g., Sponsor).
    4.  Generates a summary string for the LLM.
    5.  **Side Effect**: Injects chart data into `st.session_state` to trigger an inline chart in the UI.

    Args:
        query (str): The search query to filter studies (e.g., "cancer").
        group_by (str): The field to group by. Options: "phase", "status", "sponsor", "start_year", "condition".
        phase (Optional[str]): Optional filter for phase (e.g., "PHASE2").
        status (Optional[str]): Optional filter for status (e.g., "RECRUITING").
        sponsor (Optional[str]): Optional filter for sponsor (e.g., "Pfizer").
        intervention (Optional[str]): Optional filter for intervention (e.g., "Keytruda").

    Returns:
        str: A summary string of the top counts and a note that a chart has been generated.
    """
    index = load_index()

    # 1. Retrieve Data
    # If query is "overall" (used by the global dashboard), we fetch ALL metadata directly
    # from the underlying collection to ensure accurate counts.
    # Otherwise, we use semantic search with a high limit.
    if query.lower() == "overall":
        try:
            # Access underlying Chroma collection
            collection = index.vector_store._collection
            # Fetch all metadata (no embeddings/documents needed for analytics)
            result = collection.get(include=["metadatas"])
            data = result["metadatas"]
        except Exception as e:
            return f"Error fetching full dataset: {e}"
    else:
        # Semantic Search for specific queries (e.g., "breast cancer")
        # We fetch a larger set (1000) to get a representative sample.
        retriever = index.as_retriever(similarity_top_k=1000)
        nodes = retriever.retrieve(query)
        data = [node.metadata for node in nodes]

    df = pd.DataFrame(data)

    if df.empty:
        return "No studies found for analytics."

    # --- APPLY FILTERS (Pandas) ---
    # We apply filters here (post-retrieval) for maximum flexibility on the retrieved set.

    # Filter by Phase
    if phase:
        target_phases = [p.strip().upper().replace(" ", "") for p in phase.split(",")]
        df["phase_upper"] = df["phase"].astype(str).str.upper().str.replace(" ", "")
        mask = df["phase_upper"].apply(lambda x: any(tp in x for tp in target_phases))
        df = df[mask]

    # Filter by Status
    if status:
        df = df[df["status"].str.upper() == status.upper()]

    # Filter by Sponsor (Fuzzy match)
    if sponsor:
        target_sponsor = normalize_sponsor(sponsor).lower()
        df["org_lower"] = df["org"].astype(str).apply(normalize_sponsor).str.lower()
        df = df[df["org_lower"].str.contains(target_sponsor, regex=False)]

    # Filter by Intervention (Fuzzy match)
    if intervention:
        target_intervention = intervention.lower()
        df["intervention_lower"] = df["intervention"].astype(str).str.lower()
        df = df[df["intervention_lower"].str.contains(target_intervention, regex=False)]

    if df.empty:
        return "No studies found after applying filters."

    # Map group_by to metadata keys
    key_map = {
        "phase": "phase",
        "status": "status",
        "sponsor": "org",
        "start_year": "start_year",
        "condition": "condition",
    }

    if group_by not in key_map:
        return f"Invalid group_by field: {group_by}. Valid options: phase, status, sponsor, start_year, condition"

    col = key_map[group_by]

    # Aggregation
    if col == "start_year":
        # Ensure numeric for year
        df[col] = pd.to_numeric(df[col], errors="coerce")
        counts = df[col].value_counts().sort_index()
    elif col == "condition":
        # Split and explode to count individual conditions
        # e.g., "Diabetes, Hypertension" -> ["Diabetes", "Hypertension"]
        counts = df[col].astype(str).str.split(", ").explode().value_counts().head(10)
    else:
        # Top 10 for categorical fields
        counts = df[col].value_counts().head(10)

    summary = counts.to_string()

    # --- Generate Chart Data ---
    # This dictionary structure is expected by the frontend (Streamlit) to render Altair charts.
    chart_data = {
        "type": "bar",
        "title": f"Studies by {group_by.capitalize()}",
        "data": counts.reset_index().to_dict("records"),
        "x": "index",  # The category (e.g., Phase 1)
        "y": col,  # The count column name
    }

    # Fix for pandas value_counts name consistency
    if "index" not in chart_data["data"][0]:
        # Reset index might have named it 'index' or the column name
        # Let's standardize to ensure the UI can read it
        chart_df = counts.reset_index()
        chart_df.columns = [group_by, "Count"]
        chart_data["data"] = chart_df.to_dict("records")
        chart_data["x"] = group_by
        chart_data["y"] = "Count"

    # Store in session state for the UI to pick up
    # This allows the tool (backend) to trigger a UI update (frontend)
    if "inline_chart_data" not in st.session_state:
        st.session_state["inline_chart_data"] = chart_data
    else:
        st.session_state["inline_chart_data"] = chart_data

    return f"Found {len(df)} studies. Top counts:\n{summary}\n\n(Chart generated in UI)"


@langchain_tool("compare_studies")
def compare_studies(query: str):
    """
    Compares multiple studies or answers complex multi-part questions using query decomposition.

    Use this tool when the user asks to "compare", "contrast", or analyze differences/similarities
    between specific studies, sponsors, or phases. It breaks down the question into sub-questions.

    Args:
        query (str): The complex comparison query (e.g., "Compare the primary outcomes of Keytruda vs Opdivo").

    Returns:
        str: A detailed response synthesizing the answers to sub-questions.
    """
    index = load_index()

    # Create a base query engine for the sub-questions
    # We use a standard engine with a reasonable top_k
    base_engine = index.as_query_engine(similarity_top_k=10)

    # Wrap it in a QueryEngineTool
    query_tool = QueryEngineTool(
        query_engine=base_engine,
        metadata=ToolMetadata(
            name="clinical_trials_db",
            description="Vector database of clinical trial protocols, results, and metadata.",
        ),
    )

    # Create the SubQuestionQueryEngine
    # It will use the LLM to break the 'query' into sub-questions that target 'clinical_trials_db'
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_tool],
        use_async=True,
    )

    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error during comparison: {e}"
