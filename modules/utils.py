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
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.schema import NodeWithScore
import chromadb


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

    Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )


@st.cache_resource
def load_index() -> VectorStoreIndex:
    """
    Loads the persistent ChromaDB index.

    Uses Streamlit's @st.cache_resource to load the index only once per session.

    Returns:
        VectorStoreIndex: The loaded LlamaIndex vector store index.
    """
    setup_llama_index()
    # Initialize ChromaDB client pointing to the local persistence directory
    db = chromadb.PersistentClient(path="./ct_gov_index")
    
    # Get or create the collection for clinical trials
    chroma_collection = db.get_or_create_collection("clinical_trials")
    
    # Create the vector store wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


# --- Normalization ---
def normalize_sponsor(sponsor: str) -> Optional[str]:
    """
    Normalizes sponsor names to handle common aliases and variations.

    This is crucial for accurate filtering and aggregation, as sponsor names
    in the raw data can vary (e.g., "Merck", "MSD", "Merck Sharp & Dohme").

    Args:
        sponsor (str): The raw sponsor name.

    Returns:
        Optional[str]: The normalized canonical sponsor name, or None if input is empty.
    """
    if not sponsor:
        return None

    s = sponsor.lower().strip()
    # Mapping of common aliases to canonical names
    aliases = {
        "gsk": "GlaxoSmithKline",
        "glaxo": "GlaxoSmithKline",
        "glaxosmithkline": "GlaxoSmithKline",
        "j&j": "Janssen",
        "johnson & johnson": "Janssen",
        "johnson and johnson": "Janssen",
        "janssen": "Janssen",
        "bms": "Bristol-Myers Squibb",
        "bristol myers squibb": "Bristol-Myers Squibb",
        "merck": "Merck Sharp & Dohme",
        "msd": "Merck Sharp & Dohme",
    }

    for alias, canonical in aliases.items():
        if alias in s:
            return canonical
    return sponsor


# --- Custom Filters ---
class LocalMetadataPostFilter:
    """
    Custom LlamaIndex post-processor for filtering retrieved nodes based on metadata.

    This allows for "Post-Retrieval Filtering", where we filter the results *after*
    semantic search but *before* sending them to the LLM. This is useful for
    complex logic that vector search alone cannot handle easily (e.g., fuzzy matching).

    Attributes:
        phase (Optional[str]): Comma-separated string of phases to keep (e.g., "PHASE2,PHASE3").
        sponsor (Optional[str]): Sponsor name to filter by (fuzzy match).
    """

    def __init__(self, phase: Optional[str] = None, sponsor: Optional[str] = None):
        self.phase = phase
        self.sponsor = sponsor

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        """
        Filters the list of nodes based on the configured criteria.

        Args:
            nodes (List[NodeWithScore]): The list of nodes retrieved from the index.
            query_bundle: The query bundle (unused here, but required by signature).

        Returns:
            List[NodeWithScore]: The filtered list of nodes.
        """
        new_nodes = []
        for node in nodes:
            keep = True
            metadata = node.node.metadata

            # Phase Filter (Loose match)
            # Checks if any of the requested phases are present in the node's phase string
            if self.phase:
                node_phase = str(metadata.get("phase", "")).upper().replace(" ", "")
                target_phases = [
                    p.strip().upper().replace(" ", "") for p in self.phase.split(",")
                ]
                if not any(tp in node_phase for tp in target_phases):
                    keep = False

            # Sponsor Filter (Fuzzy/Alias match)
            # Normalizes both the target and node sponsor names before comparing
            if self.sponsor and keep:
                node_sponsor = str(metadata.get("org", ""))
                norm_target = normalize_sponsor(self.sponsor).lower()
                norm_node = normalize_sponsor(node_sponsor).lower()
                if norm_target not in norm_node:
                    keep = False

            if keep:
                new_nodes.append(node)
        return new_nodes

    def __call__(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        """Callable interface for LlamaIndex post-processors."""
        return self._postprocess_nodes(nodes, query_bundle)
