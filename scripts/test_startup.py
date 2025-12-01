
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.utils import load_index, setup_llama_index

def test_startup():
    print("🔹 Loading environment...")
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    print(f"🔹 API Key present: {bool(api_key)}")
    
    print("🔹 Setting up LlamaIndex...")
    setup_llama_index(api_key=api_key)
    
    print("🔹 Loading Index...")
    try:
        index = load_index()
        print("✅ Index loaded successfully!")
    except Exception as e:
        print(f"❌ Index load failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_startup()
