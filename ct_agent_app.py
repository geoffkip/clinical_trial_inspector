
import streamlit as st
import pandas as pd
import os
import logging
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
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

# 3. Sidebar Filters
with st.sidebar:
    st.header("🔍 Filter Studies")
    
    # Sponsor Filter
    sponsor_filter = st.text_input("Sponsor (e.g., Pfizer)")
    
    # Year Range Filter
    year_range = st.slider("Start Year Range", 2000, 2030, (2015, 2025))
    
    # Phase Filter
    phase_options = ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
    selected_phases = st.multiselect("Phase", phase_options)
    
    # Study Type Filter
    type_options = ["Interventional", "Observational", "Expanded Access"]
    selected_types = st.multiselect("Study Type", type_options)
    
    # Status Filter
    status_options = ["RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN", "ACTIVE_NOT_RECRUITING"]
    selected_statuses = st.multiselect("Recruitment Status", status_options)

# 4. Create LlamaIndex Tool
from langchain.tools import tool as langchain_tool

@langchain_tool
def search_trials(query: str):
    """
    Searches for clinical trials using semantic search.
    Returns detailed study information including Title, Status, Phase, and Summary.
    """
    # Create Query Engine with filters if possible, but for now we rely on the LLM to filter 
    # or we can implement metadata filters in LlamaIndex later.
    # For now, we pass the query to the engine.
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(query)
    return str(response)

@langchain_tool
def find_similar_studies(query: str):
    """
    Finds studies similar to a given query or study description. 
    Returns the top studies with their similarity scores.
    """
    # LlamaIndex query engine is already doing similarity search.
    # We can reuse the same logic or customize top_k
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(query)
    return str(response)

tools = [search_trials, find_similar_studies]

system_prompt = """
You are an expert Clinical Research Assistant.
1. Use 'search_trials' to find real data based on user queries.
2. Use 'find_similar_studies' when the user asks for comparisons or "similar" studies.
3. Always cite the NCT ID.
4. If the user provided filters in the context, prioritize them.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input := st.chat_input("Ex: Find Phase 3 Pfizer studies started after 2022"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Augment query with sidebar filters
    augmented_input = user_input
    filter_context = []
    if sponsor_filter:
        filter_context.append(f"Sponsor: {sponsor_filter}")
    if selected_phases:
        filter_context.append(f"Phases: {', '.join(selected_phases)}")
    if selected_types:
        filter_context.append(f"Types: {', '.join(selected_types)}")
    if selected_statuses:
        filter_context.append(f"Status: {', '.join(selected_statuses)}")
    filter_context.append(f"Year Range: {year_range[0]}-{year_range[1]}")
    
    if filter_context:
        augmented_input = f"{user_input}\n\nContext from Sidebar Filters (Apply these strictly):\n" + "\n".join(filter_context)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executor.invoke({
                    "input": augmented_input, 
                    "chat_history": st.session_state.messages
                })
                output = response["output"]
                st.markdown(output)
                st.session_state.messages.append(AIMessage(content=output))
                
                # --- ANALYTICS & EXPORT ---
                with st.expander("📊 Analytics & Export"):
                    # Use LlamaIndex retriever to fetch nodes for visualization
                    retriever = index.as_retriever(similarity_top_k=50)
                    nodes = retriever.retrieve(user_input)
                    
                    if nodes:
                        data = []
                        for node in nodes:
                            # Access metadata from the node
                            meta = node.metadata
                            data.append({
                                "NCT ID": meta.get("nct_id"),
                                "Title": meta.get("title"),
                                "Sponsor": meta.get("sponsor"),
                                "Phase": meta.get("phase"),
                                "Year": meta.get("year"),
                                "Type": meta.get("study_type")
                            })
                        
                        df = pd.DataFrame(data)
                        
                        # Visualizations
                        st.subheader("Phase Distribution")
                        st.bar_chart(df["Phase"].value_counts())
                        
                        st.subheader("Top Sponsors")
                        st.bar_chart(df["Sponsor"].value_counts().head(10))
                        
                        st.subheader("Start Year Trend")
                        if "Year" in df.columns:
                            year_counts = df["Year"].value_counts().sort_index()
                            st.line_chart(year_counts)
                        
                        # Export
                        st.subheader("📥 Export Data")
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Search Results (CSV)",
                            csv,
                            "clinical_trials_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.info("No structured data found for analytics.")

            except Exception as e:
                st.error(f"Error: {e}")