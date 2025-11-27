# Clinical Trial Detective 🕵️‍♂️💊

**Clinical Trial Detective** is an intelligent AI agent designed to help researchers, clinicians, and curious minds explore and analyze clinical trial data from [ClinicalTrials.gov](https://clinicaltrials.gov/).

Built with **LangChain**, **Streamlit**, and **Google Gemini**, this tool allows you to search for studies using natural language, filter by specific criteria, and get synthesized insights.

## ✨ Features

- **Natural Language Search**: Ask questions like "Find Phase 3 Pfizer studies for diabetes started after 2022" instead of using complex search forms.
- **Smart Filtering**: The agent automatically extracts filters (Year, Phase, Sponsor) from your query to narrow down results.
- **Local Vector Store**: Uses **ChromaDB** with **HuggingFace embeddings** (`all-MiniLM-L6-v2`) for fast, private, and cost-effective semantic search.
- **AI Synthesis**: Powered by **Google Gemini (gemini-2.5-flash)** to summarize findings, compare studies, and answer follow-up questions.
- **Real-time Data Ingestion**: Fetch the latest trials directly from the ClinicalTrials.gov API.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (`gemini-2.5-flash`)
- **Orchestration**: LangChain (Agents, Self-Query Retriever)
- **Vector Database**: ChromaDB (Local)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)

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
This script fetches the latest studies from ClinicalTrials.gov and embeds them into ChromaDB.

```bash
python ingest_ct.py
```
*Note: The first run will download the embedding model (~80MB).*

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
- *"Find studies sponsored by Pfizer in 2024."*

## 📂 Project Structure

- `ct_agent_app.py`: Main Streamlit application and agent logic.
- `ingest_ct.py`: Script to fetch data from CT.gov and build the vector database.
- `ct_gov_local_db/`: Directory where the ChromaDB vector store is persisted.
- `requirements.txt`: Python dependencies.

## ⚠️ Note on Quotas
This project uses the free tier of Google Gemini API, which has rate limits. If you encounter a "ResourceExhausted" error, please wait a minute before retrying.
