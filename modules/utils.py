"""
Utility functions for the Clinical Trial Agent.

This module handles:
1.  **Configuration**: Setting up LlamaIndex settings (LLM, Embeddings).
2.  **Index Loading**: Loading the persisted ChromaDB vector index.
3.  **Normalization**: Helper functions for standardizing data (e.g., sponsor names).
4.  **Filtering**: Custom post-processors for filtering retrieval results.
"""

import os
import streamlit as st
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.gemini import Gemini
import lancedb
from dotenv import load_dotenv

# --- MONKEYPATCH START ---
# Fix for AttributeError: 'LanceEmptyQueryBuilder' object has no attribute 'nprobes'
# AND Fix for SQL quoting bug in IN filters
original_query = LanceDBVectorStore.query

def patched_query(self, query, **kwargs):
    try:
        return original_query(self, query, **kwargs)
    except Exception as e:
        print(f"⚠️ LanceDB Query Error: {e}")
        if hasattr(query, "filters"):
            print(f"   Filters: {query.filters}")
        
        if "nprobes" in str(e):
            from llama_index.core.vector_stores.types import VectorStoreQueryResult
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        raise e

LanceDBVectorStore.query = patched_query

# Fix for SQL quoting in LanceDB filters (specifically for IN operator with strings)
from llama_index.vector_stores.lancedb import base as lancedb_base
from llama_index.core.vector_stores.types import FilterOperator

original_to_lance_filter = lancedb_base._to_lance_filter

def patched_to_lance_filter(standard_filters, metadata_keys):
    # If standard_filters is None or empty, return None
    if not standard_filters:
        return None
        
    # We need to reimplement the logic because the original function is what's broken
    # But we can't easily access the internal logic.
    # However, we can try to intercept the result? No, it returns a string (SQL where clause).
    
    # Let's try to reimplement a robust version for IN operator
    filters = []
    for filter in standard_filters.filters:
        key = filter.key
        if metadata_keys and key not in metadata_keys:
             continue
        
        # LanceDB stores metadata in a struct column named 'metadata'
        # So we must prefix the key
        lance_key = f"metadata.{key}"
             
        # Handle IN operator specifically to fix quoting
        if filter.operator == FilterOperator.IN:
            if isinstance(filter.value, list):
                # Quote strings properly
                values = []
                for v in filter.value:
                    if isinstance(v, str):
                        values.append(f"'{v}'") # Single quotes for SQL
                    else:
                        values.append(str(v))
                val_str = ", ".join(values)
                filters.append(f"{lance_key} IN ({val_str})")
                continue
        
        # Fallback to original logic for other operators (or we'd have to reimplement all)
        # But we can't mix our string with the original function's result easily if we call it per filter.
        # The original function iterates over ALL filters and joins them.
        
        # So we MUST reimplement the whole function or at least the loop.
        # Basic implementation based on common LlamaIndex patterns:
        op = filter.operator
        val = filter.value
        
        if op == FilterOperator.EQ:
            if isinstance(val, str):
                filters.append(f"{lance_key} = '{val}'")
            else:
                filters.append(f"{lance_key} = {val}")
        elif op == FilterOperator.GT:
            filters.append(f"{lance_key} > {val}")
        elif op == FilterOperator.LT:
            filters.append(f"{lance_key} < {val}")
        elif op == FilterOperator.GTE:
            filters.append(f"{lance_key} >= {val}")
        elif op == FilterOperator.LTE:
            filters.append(f"{lance_key} <= {val}")
        elif op == FilterOperator.NE:
            if isinstance(val, str):
                filters.append(f"{lance_key} != '{val}'")
            else:
                filters.append(f"{lance_key} != {val}")
        # Add other operators as needed
        
    if not filters:
        return None
        
    return " AND ".join(filters)

lancedb_base._to_lance_filter = patched_to_lance_filter
# --- MONKEYPATCH END ---




def load_environment():
    """Loads environment variables from .env file."""
    load_dotenv()


# --- Configuration ---
def setup_llama_index():
    """
    Configures the global LlamaIndex settings.

    Sets up:
    - **LLM**: Google Gemini (gemini-2.5-flash) with temperature=0 for deterministic outputs.
    - **Embeddings**: PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) for biomedical domain specificity.

    Raises:
        SystemExit: If GOOGLE_API_KEY is not found in environment variables.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("Please set GOOGLE_API_KEY in .env")
        st.stop()

    try:
        Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0)
    except Exception as e:
        print(f"⚠️ LLM initialization failed (likely missing API key): {e}")
        print("⚠️ Using MockLLM for testing/fallback.")
        from llama_index.core.llms import MockLLM
        Settings.llm = MockLLM()
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )


@st.cache_resource
def load_index() -> VectorStoreIndex:
    """
    Loads the persistent LanceDB index.

    Uses Streamlit's @st.cache_resource to load the index only once per session.

    Returns:
        VectorStoreIndex: The loaded LlamaIndex vector store index.
    """
    setup_llama_index()
    
    # Initialize LanceDB
    db_path = "./ct_gov_lancedb"
    db = lancedb.connect(db_path)

    # Define metadata keys explicitly to ensure filters work
    # This prevents the "NoneType is not iterable" error in _to_lance_filter
    metadata_keys = [
        "nct_id", "title", "org", "sponsor", "status", "phase", 
        "study_type", "start_year", "condition", "intervention", 
        "country", "state"
    ]

    # Create the vector store wrapper
    vector_store = LanceDBVectorStore(
        uri=db_path, 
        table_name="clinical_trials",
        query_mode="hybrid",
    )
    
    # Manually set metadata keys since the constructor doesn't accept them
    # and automatic inference might fail on read-only/empty connections
    vector_store._metadata_keys = metadata_keys

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def get_hybrid_retriever(index: VectorStoreIndex, similarity_top_k: int = 50, filters=None):
    """
    Creates a Hybrid Retriever using LanceDB's native hybrid search.
    
    Args:
        index (VectorStoreIndex): The loaded vector index.
        similarity_top_k (int): Number of top results to retrieve.
        filters (MetadataFilters, optional): Filters to apply.
        
    Returns:
        VectorIndexRetriever: The configured retriever.
    """
    # LanceDB supports native hybrid search via query_mode="hybrid"
    # We pass this configuration to the retriever
    # Use standard retriever first to avoid LanceDB hybrid search issues on small datasets
    return index.as_retriever(
        similarity_top_k=similarity_top_k, 
        filters=filters,
    )


# --- Normalization ---

# Centralized Sponsor Mappings
# Key: Canonical Name
# Value: List of variations/aliases (including the canonical name itself if needed for matching)
SPONSOR_MAPPINGS = {
    "GlaxoSmithKline": [
        "gsk", "glaxo", "glaxosmithkline", "glaxosmithkline", 
        "GlaxoSmithKline"
    ],
    "Janssen": [
        "j&j", "johnson & johnson", "johnson and johnson", "janssen", "Janssen",
        "Janssen Research & Development, LLC",
        "Janssen Vaccines & Prevention B.V.",
        "Janssen Pharmaceutical K.K.",
        "Janssen-Cilag International NV",
        "Janssen Sciences Ireland UC",
        "Janssen Pharmaceutica N.V., Belgium",
        "Janssen Scientific Affairs, LLC",
        "Janssen-Cilag Ltd.",
        "Xian-Janssen Pharmaceutical Ltd.",
        "Janssen Korea, Ltd., Korea",
        "Janssen-Cilag G.m.b.H",
        "Janssen-Cilag, S.A.",
        "Janssen BioPharma, Inc.",
    ],
    "Bristol-Myers Squibb": [
        "bms", "bristol", "bristol myers squibb", "bristol-myers squibb",
        "Bristol-Myers Squibb"
    ],
    "Merck Sharp & Dohme": [
        "merck", "msd", "merck sharp & dohme", 
        "Merck Sharp & Dohme LLC"
    ],
    "Pfizer": ["pfizer", "Pfizer", "Pfizer Inc."],
    "AstraZeneca": ["astrazeneca", "AstraZeneca"],
    "Eli Lilly and Company": ["lilly", "eli lilly", "Eli Lilly and Company"],
    "Sanofi": ["sanofi", "Sanofi"],
    "Novartis": ["novartis", "Novartis"],
}

def normalize_sponsor(sponsor: str) -> Optional[str]:
    """
    Normalizes sponsor names to handle common aliases and variations.
    
    Uses SPONSOR_MAPPINGS to map aliases (e.g., "J&J") and specific variations 
    (e.g., "Janssen Research & Development, LLC") to the canonical name ("Janssen").

    Args:
        sponsor (str): The raw sponsor name.

    Returns:
        Optional[str]: The normalized canonical sponsor name, or None if input is empty.
    """
    if not sponsor:
        return None

    s = sponsor.lower().strip()
    
    for canonical, variations in SPONSOR_MAPPINGS.items():
        # Check if input matches canonical name (case-insensitive)
        if s == canonical.lower():
            return canonical
            
        # Check if input matches any variation (substring or exact match?)
        # For aliases like "gsk", substring match is risky (e.g. "gsk" in "gskill").
        # But for "Janssen Research...", we want to map it to "Janssen".
        
        # Strategy:
        # 1. Check exact match against variations
        # 2. Check if alias is a substring of input (for short aliases like "gsk")
        
        for v in variations:
            v_lower = v.lower()
            if v_lower == s:
                return canonical
            # If the variation is a known alias (like 'gsk'), check if it's in the string
            if len(v) < 5 and v_lower in s: 
                 return canonical
            # If the input is a variation (e.g. input="Janssen Research...", variation="Janssen Research...")
            # This is covered by exact match above.
            
            # What if input is "Janssen Research" and variation is "Janssen Research & Development"?
            # We might want to check if canonical is in the input?
            if canonical.lower() in s:
                return canonical

    return sponsor


def get_sponsor_variations(sponsor: str) -> Optional[List[str]]:
    """
    Returns a list of exact database 'org' values for a given sponsor alias.
    This enables strict pre-filtering using the IN operator.
    """
    if not sponsor:
        return None

    # First, normalize the input to get the canonical name
    canonical = normalize_sponsor(sponsor)
    
    # If we have a mapping for this canonical name, return the variations
    # BUT, we only want the "official" DB variations, not the short aliases (like "gsk").
    # The original logic had specific lists for DB values.
    # We should probably separate "Aliases" from "DB Values" in the mapping if we want to be precise.
    # Or just return all of them? 
    # If we return "gsk" in the IN clause, it won't match anything in the DB (which has full names), 
    # but it doesn't hurt.
    
    if canonical in SPONSOR_MAPPINGS:
        return SPONSOR_MAPPINGS[canonical]
        
    return None





