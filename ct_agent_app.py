
import streamlit as st
import pandas as pd
import os
import logging
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
import chromadb

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Clinical Trial Inspector", layout="wide")
st.title("🧬 Clinical Trial Inspector Agent")

# 1. Setup LLM (Gemini) & Embeddings (Local)


# 1. Setup LLM
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set GOOGLE_API_KEY in .env")
    st.stop()

# Configure LlamaIndex to use Gemini
Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Load LlamaIndex
@st.cache_resource
def load_index():
    print("🧠 Loading LlamaIndex...")
    # Initialize Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    
    # Initialize ChromaDB Client
    db = chromadb.PersistentClient(path="./ct_gov_index")
    chroma_collection = db.get_or_create_collection("clinical_trials")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index

index = load_index()

# 3. Create LlamaIndex Tool
from langchain.tools import tool as langchain_tool

class LocalMetadataPostFilter(BaseNodePostprocessor):
    phase: Optional[str] = None
    sponsor: Optional[str] = None
    
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        filtered_nodes = []
        for node in nodes:
            meta = node.metadata
            match = True
            if self.phase:
                # Check if phase is in the metadata string (case-insensitive)
                if self.phase.lower() not in meta.get("phase", "").lower():
                    match = False
            if self.sponsor:
                if self.sponsor.lower() not in meta.get("org", "").lower():
                    match = False
            
            if match:
                filtered_nodes.append(node)
        return filtered_nodes

@langchain_tool
def search_trials(query: str, status: str = None, phase: str = None, sponsor: str = None, year: int = None):
    """
    Searches for clinical trials using semantic search with optional strict filters.
    
    Args:
        query: The search query (e.g., "diabetes treatment").
        status: Filter by status (e.g., "RECRUITING", "COMPLETED").
        phase: Filter by phase (e.g., "PHASE2", "PHASE3").
        sponsor: Filter by sponsor name (e.g., "Pfizer").
        year: Filter for studies starting after this year (e.g., 2020).
    """
    # 1. Pre-retrieval filters (supported by Chroma)
    filters = []
    if status:
        filters.append(MetadataFilter(key="status", value=status.upper(), operator=FilterOperator.EQ))
    if year:
        filters.append(MetadataFilter(key="start_year", value=year, operator=FilterOperator.GTE))
        
    metadata_filters = MetadataFilters(filters=filters) if filters else None
    
    # 2. Post-retrieval filters (custom logic)
    post_filters = []
    if phase or sponsor:
        post_filters.append(LocalMetadataPostFilter(phase=phase, sponsor=sponsor))
    
    # 3. Re-ranker
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)
    post_filters.append(reranker)
    
    query_engine = index.as_query_engine(
        similarity_top_k=20, # Fetch more to allow for filtering
        node_postprocessors=post_filters,
        filters=metadata_filters
    )
    response = query_engine.query(query)
    return str(response)

@langchain_tool
def find_similar_studies(query: str):
    """
    Finds studies similar to a given query or study description. 
    Returns the top studies with their similarity scores.
    """
    # Initialize Re-ranker
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)
    
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker]
    )
    response = query_engine.query(f"Find studies similar to: {query}")
    return str(response)

# 4. Define Agent
tools = [search_trials, find_similar_studies]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Clinical Trial Expert Assistant. Use the `search_trials` tool to find relevant studies based on the user's query. "
            "Use `find_similar_studies` when the user asks for similar trials. "
            "Always provide the NCT ID, Title, Status, and a brief summary for each study found. "
            "If the user asks for a comparison, search for the relevant studies first and then compare them based on the retrieved data."
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about clinical trials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing clinical trials..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                output = response["output"]
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# 6. Analytics & Export (On-Demand)
with st.expander("📊 Analytics & Export"):
    st.write("Analyze the top 50 most relevant studies to your last query (or general trends).")
    
    if st.button("Load Analytics"):
        with st.spinner("Generating analytics..."):
            # Use LlamaIndex Retriever to fetch nodes for analytics
            # We fetch a larger batch (e.g., 50) to generate meaningful charts
            retriever = index.as_retriever(similarity_top_k=50)
            nodes = retriever.retrieve("clinical trials") # Retrieve general or context-based nodes
            
            data = []
            for node in nodes:
                meta = node.metadata
                data.append({
                    "Phase": meta.get("phase", "NA"),
                    "Sponsor": meta.get("org", "Unknown"),
                    "Year": meta.get("start_year", 0),
                    "Status": meta.get("status", "Unknown"),
                    "Study Type": meta.get("study_type", "Unknown"),
                    "Country": meta.get("country", "Unknown")
                })
            
            if data:
                df = pd.DataFrame(data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Phase Distribution")
                    phase_counts = df['Phase'].value_counts()
                    st.bar_chart(phase_counts)
                    
                with col2:
                    st.subheader("Top Sponsors")
                    sponsor_counts = df['Sponsor'].value_counts().head(10)
                    st.bar_chart(sponsor_counts)
                    
                st.subheader("Start Year Trend")
                year_counts = df['Year'].value_counts().sort_index()
                st.line_chart(year_counts)
                
                # Export Button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Search Results (CSV)",
                    csv,
                    "clinical_trials_results.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.info("No data available for analytics.")