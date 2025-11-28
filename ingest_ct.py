import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import argparse

# LlamaIndex Imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

import re

# Disable LLM for ingestion (we only need embeddings)
Settings.llm = None

def clean_text(text):
    """Removes HTML tags and extra whitespace."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_trials_generator(years=5, max_studies=1000, status=None, phases=None):
    """
    Yields batches of clinical trials from the last 'years' years using pagination.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    print(f"📡 Connecting to CT.gov API...")
    print(f"🔎 Fetching trials starting after: {start_date}")
    if status:
        print(f"   Filters - Status: {status}")
    if phases:
        print(f"   Filters - Phases: {phases}")

    fetched_count = 0
    next_page_token = None
    
    # If max_studies is -1, fetch ALL studies
    fetch_limit = float('inf') if max_studies == -1 else max_studies

    while fetched_count < fetch_limit:
        current_limit = 1000
        if max_studies != -1:
             current_limit = min(1000, max_studies - fetched_count)

        # Construct query.term with filters
        query_parts = [f"AREA[StartDate]RANGE[{start_date},MAX]"]
        
        if status:
            status_str = " OR ".join(status)
            query_parts.append(f"AREA[OverallStatus]({status_str})")
            
        if phases:
            phase_str = " OR ".join(phases)
            query_parts.append(f"AREA[Phase]({phase_str})")
            
        full_query = " AND ".join(query_parts)

        params = {
            "query.term": full_query,
            "pageSize": current_limit,
            "fields": "protocolSection.identificationModule.nctId,protocolSection.identificationModule.briefTitle,protocolSection.identificationModule.officialTitle,protocolSection.identificationModule.organization,protocolSection.statusModule.overallStatus,protocolSection.statusModule.startDateStruct,protocolSection.statusModule.completionDateStruct,protocolSection.designModule.phases,protocolSection.designModule.studyType,protocolSection.eligibilityModule.eligibilityCriteria,protocolSection.eligibilityModule.sex,protocolSection.eligibilityModule.stdAges,protocolSection.descriptionModule.briefSummary,protocolSection.conditionsModule.conditions,protocolSection.outcomesModule.primaryOutcomes,protocolSection.contactsLocationsModule.locations"
        }
        
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                studies = data.get('studies', [])
                
                if not studies:
                    break
                
                yield studies
                
                fetched_count += len(studies)
                next_page_token = data.get('nextPageToken')
                print(f"   Fetched batch of {len(studies)}. Total: {fetched_count}")
                
                if not next_page_token:
                    print("   No more pages available.")
                    break
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                break
        except Exception as e:
            print(f"❌ Request Error: {e}")
            break

def run_ingestion():
    parser = argparse.ArgumentParser(description="Ingest Clinical Trials data.")
    parser.add_argument("--limit", type=int, default=2000, help="Number of studies to ingest. Set to -1 for ALL.")
    parser.add_argument("--years", type=int, default=5, help="Number of years to look back.")
    parser.add_argument("--status", type=str, default="COMPLETED", help="Comma-separated list of statuses (e.g., COMPLETED,RECRUITING).")
    parser.add_argument("--phases", type=str, default="PHASE2,PHASE3", help="Comma-separated list of phases (e.g., PHASE2,PHASE3).")
    args = parser.parse_args()

    status_list = args.status.split(',') if args.status else []
    phase_list = args.phases.split(',') if args.phases else []

    print(f"⚙️ Configuration: Limit={args.limit}, Years={args.years}")
    print(f"   Status Filter: {status_list}")
    print(f"   Phase Filter: {phase_list}")
    
    # --- INITIALIZE LLAMAINDEX COMPONENTS ---
    print("🧠 Initializing LlamaIndex Embeddings (PubMedBERT)...")
    embed_model = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    
    # Initialize ChromaDB
    print("🚀 Initializing ChromaDB...")
    db = chromadb.PersistentClient(path="./ct_gov_index")
    chroma_collection = db.get_or_create_collection("clinical_trials")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    total_ingested = 0
    
    # Create Index (initially empty or load existing if we wanted, but here we are ingesting fresh)
    # We will add documents to the index in batches
    
    for batch_studies in fetch_trials_generator(years=args.years, max_studies=args.limit, status=status_list, phases=phase_list):
        documents = []
        for study in batch_studies:
            try:
                protocol = study.get('protocolSection', {})
                identification = protocol.get('identificationModule', {})
                status_module = protocol.get('statusModule', {})
                design = protocol.get('designModule', {})
                eligibility = protocol.get('eligibilityModule', {})
                description = protocol.get('descriptionModule', {})
                conditions_module = protocol.get('conditionsModule', {})
                outcomes_module = protocol.get('outcomesModule', {})
                locations_module = protocol.get('contactsLocationsModule', {})

                nct_id = identification.get('nctId', 'N/A')
                title = identification.get('briefTitle', 'N/A')
                official_title = identification.get('officialTitle', 'N/A')
                org = identification.get('organization', {}).get('fullName', 'N/A')
                summary = clean_text(description.get('briefSummary', 'N/A'))
                
                overall_status = status_module.get('overallStatus', 'N/A')
                start_date = status_module.get('startDateStruct', {}).get('date', 'N/A')
                completion_date = status_module.get('completionDateStruct', {}).get('date', 'N/A')
                
                phases = ", ".join(design.get('phases', []))
                study_type = design.get('studyType', 'N/A')
                
                criteria = clean_text(eligibility.get('eligibilityCriteria', 'N/A'))
                gender = eligibility.get('sex', 'N/A')
                ages = ", ".join(eligibility.get('stdAges', []))
                
                conditions = ", ".join(conditions_module.get('conditions', []))
                
                primary_outcomes = []
                for outcome in outcomes_module.get('primaryOutcomes', []):
                    measure = outcome.get('measure', '')
                    desc = outcome.get('description', '')
                    primary_outcomes.append(f"- {measure}: {desc}")
                outcomes_str = clean_text("\n".join(primary_outcomes))

                locations = []
                for loc in locations_module.get('locations', []):
                    facility = loc.get('facility', 'N/A')
                    city = loc.get('city', '')
                    country = loc.get('country', '')
                    locations.append(f"{facility} ({city}, {country})")
                locations_str = "; ".join(locations[:5]) # Limit to 5 locations to save space

                # Construct Rich Page Content with Markdown Headers
                page_content = (
                    f"# {title}\n"
                    f"**NCT ID:** {nct_id}\n"
                    f"**Official Title:** {official_title}\n"
                    f"**Sponsor:** {org}\n"
                    f"**Status:** {overall_status}\n"
                    f"**Phase:** {phases}\n"
                    f"**Study Type:** {study_type}\n"
                    f"**Start Date:** {start_date}\n"
                    f"**Completion Date:** {completion_date}\n\n"
                    f"## Summary\n{summary}\n\n"
                    f"## Conditions\n{conditions}\n\n"
                    f"## Eligibility Criteria\n"
                    f"**Gender:** {gender}\n"
                    f"**Ages:** {ages}\n"
                    f"**Criteria:**\n{criteria}\n\n"
                    f"## Primary Outcomes\n{outcomes_str}\n\n"
                    f"## Locations\n{locations_str}"
                )
                
                # Metadata for filtering
                metadata = {
                    "nct_id": nct_id,
                    "title": title,
                    "org": org,
                    "status": overall_status,
                    "phase": phases,
                    "study_type": study_type,
                    "start_year": int(start_date.split('-')[0]) if start_date != 'N/A' else 0,
                    "condition": conditions,
                    "country": locations[0].split(',')[-1].strip() if locations else "Unknown"
                }
                
                doc = Document(
                    text=page_content,
                    metadata=metadata,
                    id_=nct_id 
                )
                documents.append(doc)
            except Exception as e:
                print(f"⚠️ Error processing study {study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'Unknown')}: {e}")
                continue
        
        if documents:
            # Create Index from documents (this persists to Chroma automatically via StorageContext)
            # Note: VectorStoreIndex.from_documents creates a NEW index. 
            # To append, we should use an existing index or just insert nodes.
            # For simplicity in this script, we'll use from_documents with the storage context
            # which pushes to Chroma.
            
            VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                embed_model=embed_model
            )
            
            total_ingested += len(documents)
            print(f"   ✅ Persisted {len(documents)} docs. Total Ingested: {total_ingested}")

    print(f"🎉 Ingestion Complete! Total studies in DB: {total_ingested}")

if __name__ == "__main__":
    run_ingestion()