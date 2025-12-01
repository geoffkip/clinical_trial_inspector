
import time
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.tools import search_trials
from modules.utils import setup_llama_index

def test_rag_performance():
    load_dotenv()
    
    # Ensure LLM is set up (needed for expand_query if used, though search_trials handles it)
    setup_llama_index()
    
    query = "immunotherapy for lung cancer"
    print(f"🚀 Starting RAG Search for: '{query}'")
    
    start_time = time.time()
    try:
        # LangChain tools must be invoked with a dict
        results = search_trials.invoke({"query": query})
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ Search completed in {duration:.2f} seconds.")
        print(f"📄 Result length: {len(results)} chars")
        print("--- Preview ---")
        print(results[:500] + "...")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_performance()
