import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# We use Local Embeddings (HuggingFace) to avoid Google API Rate Limits
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

def fetch_trials(years=5, max_studies=10000):
    """
    Fetches clinical trials from the last 'years' years using pagination.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    print(f"📡 Connecting to CT.gov API...")
    print(f"🔎 Fetching trials starting after: {start_date}")

    all_studies = []
    next_page_token = None
    
    while len(all_studies) < max_studies:
        params = {
            "query.term": f"AREA[StartDate]RANGE[{start_date},MAX]",
            "pageSize": min(1000, max_studies - len(all_studies)), # Adjust page size to not over-fetch
            "fields": "protocolSection.identificationModule.nctId,protocolSection.identificationModule.briefTitle,protocolSection.eligibilityModule.eligibilityCriteria,protocolSection.designModule.phases,protocolSection.statusModule.startDateStruct,protocolSection.identificationModule.organization"
        }
        
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                studies = data.get('studies', [])
                all_studies.extend(studies)
                
                next_page_token = data.get('nextPageToken')
                print(f"   Fetched {len(studies)} studies. Total: {len(all_studies)}")
                
                if not next_page_token:
                    print("   No more pages available.")
                    break
            else:
                print(f"❌ API Error: {response.status_code}")
                break
        except Exception as e:
            print(f"❌ Request Error: {e}")
            break
            
    return all_studies[:max_studies]

def run_ingestion():
    # Set a higher default limit or let user configure it
    studies = fetch_trials(years=5, max_studies=10000) 
    
    if not studies:
        print("No studies found.")
        return

    documents = []
    print(f"⚙️ Processing {len(studies)} studies...")

    for study in studies:
        try:
            protocol = study.get('protocolSection', {})
            ident = protocol.get('identificationModule', {})
            design = protocol.get('designModule', {})
            status = protocol.get('statusModule', {})
            
            nct_id = ident.get('nctId', 'Unknown')
            title = ident.get('briefTitle', 'Untitled')
            criteria = protocol.get('eligibilityModule', {}).get('eligibilityCriteria', 'No criteria provided.')
            sponsor = ident.get('organization', {}).get('fullName', 'Unknown')
            phases = design.get('phases', [])
            phase_str = phases[0] if phases else "NA"
            
            start_date_struct = status.get('startDateStruct', {})
            date_str = start_date_struct.get('date', '1900-01-01')
            start_year = int(date_str.split('-')[0]) if '-' in date_str else 1900

            doc = Document(
                page_content=f"Study ID: {nct_id}\nTitle: {title}\nSponsor: {sponsor}\nPhase: {phase_str}\n\nCriteria Details:\n{criteria}",
                metadata={
                    "nct_id": nct_id,
                    "title": title,
                    "phase": phase_str,
                    "year": start_year,
                    "sponsor": sponsor
                }
            )
            documents.append(doc)
        except Exception:
            continue

    # --- THE FIX: USE LOCAL EMBEDDINGS ---
    print("🧠 Initializing Local Embeddings (PubMedBERT)...")
    print("(This involves a one-time download)")
    # Using a model better suited for clinical text
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    
    print("🚀 Ingesting into ChromaDB...")
    # Since this is local, we don't need batching or sleep timers!
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory="./ct_gov_local_db" 
    )
    print(f"✅ Success! Database created at ./ct_gov_local_db with {len(documents)} trials.")

if __name__ == "__main__":
    run_ingestion()