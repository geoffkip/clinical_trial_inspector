# Clinical Trial Detective 🕵️‍♂️💊

**Clinical Trial Detective** is an intelligent AI agent designed to help researchers, clinicians, and curious minds explore and analyze clinical trial data from [ClinicalTrials.gov](https://clinicaltrials.gov/).

Built with **LangChain**, **LlamaIndex**, **Streamlit**, and **Google Gemini**, this tool allows you to search for studies using natural language, filter by specific criteria, visualize trends, and get synthesized insights.

## ✨ Features

- **Natural Language Search**: Ask questions like "Find Phase 3 Pfizer studies for diabetes started after 2022" instead of using complex search forms.
- **Advanced RAG Pipeline**: Powered by **LlamaIndex** for robust document indexing and retrieval.
- **Smart Filtering**: The agent uses a hybrid approach (database + post-processing) to strictly filter results by **Year**, **Phase**, **Sponsor**, and **Status** based on your natural language query.
- **Analytics Dashboard**: Visualize search results with charts for **Phase Distribution**, **Top Sponsors**, and **Yearly Trends**.
- **Data Export**: Download your search results as a CSV file for further analysis.
- **Comprehensive Data**: Ingests detailed study information including **Summaries**, **Outcomes**, **Eligibility Criteria**, **Conditions**, **Locations**, and **Study Population**.
- **Local Vector Store**: Uses **ChromaDB** with **PubMedBERT** embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`) for clinical-specific semantic search.
- **AI Synthesis**: Powered by **Google Gemini (gemini-2.5-flash)** to summarize findings, compare studies, and answer follow-up questions.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (`gemini-2.5-flash`)
- **Orchestration**: LangChain (Agents, Tool Calling)
- **Retrieval (RAG)**: LlamaIndex (VectorStoreIndex, QueryEngine)
- **Vector Database**: ChromaDB (Local)
- **Embeddings**: HuggingFace (`pritamdeka/S-PubMedBert-MS-MARCO`)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Cloud API Key with access to Gemini

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd clinical_trial_agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**
   Create a `.env` file in the root directory and add your Google API Key:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 📖 Usage

### 1. Ingest Data
Before running the agent, you need to populate the local database with clinical trial data.
This script fetches the latest studies from ClinicalTrials.gov and embeds them into ChromaDB using LlamaIndex.

```bash
# Ingest 2000 studies from the last 5 years (default: COMPLETED, PHASE2/3)
python ingest_ct.py

# Ingest with custom status and phases
python ingest_ct.py --status RECRUITING,COMPLETED --phases PHASE3

# Ingest 5000 studies from the last 10 years
python ingest_ct.py --limit 5000 --years 10

# Ingest ALL studies (Warning: Takes a long time!)
python ingest_ct.py --limit -1
```
*Note: The first run will download the embedding model (~400MB).*

### 2. Run the Agent
Launch the Streamlit application:

```bash
streamlit run ct_agent_app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Ask Questions!
Try queries like:
- *"What are the inclusion criteria for recent Moderna trials?"*
- *"Compare the Phase 3 studies for lung cancer."*
- *"Find recruiting studies for Alzheimer's in the US."*

## 📂 Project Structure

- `ct_agent_app.py`: Main Streamlit application, LangChain agent, and LlamaIndex retrieval logic.
- `ingest_ct.py`: Script to fetch data from CT.gov and build the LlamaIndex vector store.
- `ct_gov_index/`: Directory where the LlamaIndex/ChromaDB data is persisted.
- `requirements.txt`: Python dependencies.

## ⚠️ Note on Quotas
This project uses the free tier of Google Gemini API, which has rate limits. If you encounter a "ResourceExhausted" error, please wait a minute before retrying.
