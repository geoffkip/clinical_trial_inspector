"""
Script to remove duplicate NCT IDs from the ChromaDB collection.

This script scans the 'clinical_trials' collection, identifies records with duplicate 'nct_id' metadata,
and removes the extras, keeping only one instance per NCT ID.
"""
import chromadb
import os
from collections import defaultdict

def remove_duplicates():
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, "ct_gov_index")
    
    if not os.path.exists(db_path):
        print(f"❌ Database directory '{db_path}' does not exist.")
        return

    print(f"📂 Loading database from {db_path}...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("clinical_trials")
        
        print("🔍 Scanning for duplicates...")
        # Fetch all IDs and metadata
        data = collection.get(include=["metadatas"])
        
        ids = data["ids"]
        metadatas = data["metadatas"]
        
        if not ids:
            print("Database is empty.")
            return

        # Map NCT ID -> List of Chroma IDs
        nct_map = defaultdict(list)
        for i, meta in enumerate(metadatas):
            nct_id = meta.get("nct_id")
            if nct_id:
                nct_map[nct_id].append(ids[i])
        
        # Identify duplicates
        duplicates = {k: v for k, v in nct_map.items() if len(v) > 1}
        
        if not duplicates:
            print("✅ No duplicates found. Database is clean.")
            return
            
        print(f"⚠️ Found {len(duplicates)} NCT IDs with duplicate records.")
        
        ids_to_delete = []
        for nct_id, chroma_ids in duplicates.items():
            # Keep the first one, delete the rest
            # In a more advanced version, we could check which one has more data, 
            # but usually they are identical.
            print(f"   - {nct_id}: Found {len(chroma_ids)} copies. Removing {len(chroma_ids) - 1}.")
            ids_to_delete.extend(chroma_ids[1:])
            
        if ids_to_delete:
            print(f"🗑️ Deleting {len(ids_to_delete)} duplicate records...")
            collection.delete(ids=ids_to_delete)
            print("🎉 Deduplication complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    remove_duplicates()
