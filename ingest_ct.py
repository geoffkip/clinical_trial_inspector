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

# Disable LLM for ingestion (we only need embeddings)
Settings.llm = None

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
                ident = protocol.get('identificationModule', {})
                design = protocol.get('designModule', {})
                status_mod = protocol.get('statusModule', {})
                eligibility = protocol.get('eligibilityModule', {})
                desc_mod = protocol.get('descriptionModule', {})
                cond_mod = protocol.get('conditionsModule', {})
                outcomes_mod = protocol.get('outcomesModule', {})
                loc_mod = protocol.get('contactsLocationsModule', {})
                
                # Basic Info
                nct_id = ident.get('nctId', 'Unknown')
                brief_title = ident.get('briefTitle', 'Untitled')
                official_title = ident.get('officialTitle', 'No Official Title')
                sponsor = ident.get('organization', {}).get('fullName', 'Unknown')
                
                # Status & Dates
                overall_status = status_mod.get('overallStatus', 'Unknown')
                start_date_struct = status_mod.get('startDateStruct', {})
                start_date = start_date_struct.get('date', 'Unknown')
                completion_date_struct = status_mod.get('completionDateStruct', {})
                completion_date = completion_date_struct.get('date', 'Unknown')
                
                start_year = int(start_date.split('-')[0]) if '-' in start_date else 1900

                # Design
                phases = design.get('phases', [])
                phase_str = phases[0] if phases else "NA"
                study_type = design.get('studyType', 'Unknown')
                
                # Details
                summary = desc_mod.get('briefSummary', 'No summary provided.')
                criteria = eligibility.get('eligibilityCriteria', 'No criteria provided.')
                sex = eligibility.get('sex', 'All')
                std_ages = eligibility.get('stdAges', [])
                age_str = ", ".join(std_ages) if std_ages else "Not specified"
                
                conditions = cond_mod.get('conditions', [])
                condition_str = ", ".join(conditions)
                primary_condition = conditions[0] if conditions else "Unknown"
                
                primary_outcomes = outcomes_mod.get('primaryOutcomes', [])
                outcomes_str = "; ".join([f"{o.get('measure', '')}: {o.get('timeFrame', '')}" for o in primary_outcomes])
                
                locations = loc_mod.get('locations', [])
                countries = list(set([loc.get('country', '') for loc in locations if loc.get('country')]))
                country_str = ", ".join(countries)
                primary_country = countries[0] if countries else "Unknown"

                # Construct Rich Content
                text_content = (
                    f"Study ID: {nct_id}\n"
                    f"Title: {brief_title}\n"
                    f"Official Title: {official_title}\n"
                    f"Sponsor: {sponsor}\n"
                    f"Status: {overall_status} (Start: {start_date}, End: {completion_date})\n"
                    f"Phase: {phase_str}\n"
                    f"Type: {study_type}\n"
                    f"Conditions: {condition_str}\n"
                    f"Population: Sex: {sex}, Ages: {age_str}\n"
                    f"Locations: {country_str}\n\n"
                    f"Summary:\n{summary}\n\n"
                    f"Primary Outcomes:\n{outcomes_str}\n\n"
                    f"Eligibility Criteria:\n{criteria}"
                )

                doc = Document(
                    text=text_content,
                    metadata={
                        "nct_id": nct_id,
                        "title": brief_title,
                        "phase": phase_str,
                        "study_type": study_type,
                        "year": start_year,
                        "sponsor": sponsor,
                        "status": overall_status,
                        "condition": primary_condition,
                        "country": primary_country
                    }
                )
                documents.append(doc)
            except Exception:
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