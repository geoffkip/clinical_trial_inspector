import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.utils import load_environment, get_hybrid_retriever, setup_llama_index
from modules.tools import get_study_details

# Load Env
load_environment()
setup_llama_index()

# Initialize LLM & Embeddings for Ragas
# Ragas uses LangChain LLMs/Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define Golden Dataset
# (Question, Ground Truth)
golden_dataset = [
    {
        "question": "What is the primary outcome of the study NCT04589845?",
        "ground_truth": "The primary outcome is the Overall Response Rate (ORR) as assessed by the Independent Review Committee (IRC)."
    },
    {
        "question": "Which sponsor is conducting the trial for Teclistamab in Multiple Myeloma?",
        "ground_truth": "Janssen Research & Development, LLC is the sponsor."
    },
    {
        "question": "What are the inclusion criteria regarding age for NCT04589845?",
        "ground_truth": "Patients must be 18 years of age or older."
    }
]

def run_rag_pipeline(question):
    """
    Simulates the RAG pipeline: Retrieve -> Generate
    Returns (answer, contexts)
    """
    # Load index (cached if possible, but here we just call it)
    from modules.utils import load_index
    # Use absolute path to project root's ct_gov_lancedb
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ct_gov_lancedb"))
    index = load_index(persist_dir=db_path)
    retriever = get_hybrid_retriever(index)
    # Configure top_k on the retriever if possible, or just use default
    # LanceDB retriever usually defaults to top_k=10 or similar
    # If we need to change it, we might need to modify get_hybrid_retriever to accept it
    # or access the underlying property.
    # For now, let's just call it without args as per signature.
    nodes = retriever.retrieve(question)
    nodes = retriever.retrieve(question)
    contexts = [node.get_content() for node in nodes]
    
    # 2. Generate
    # Simple generation using the retrieved context
    context_str = "\n\n".join(contexts)
    prompt = f"""
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    response = llm.invoke(prompt)
    answer = response.content
    
    return answer, contexts

def main():
    print("🚀 Starting RAG Evaluation with Ragas...")
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # Run Pipeline
    for item in golden_dataset:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"Processing: {q}")
        answer, contexts = run_rag_pipeline(q)
        
        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(gt)
        
    # Create Dataset
    dataset = Dataset.from_dict(data)
    
    # Evaluate
    # Pass Gemini LLM/Embeddings to metrics
    # Note: Ragas metrics allow passing 'llm' and 'embeddings'
    
    # Configure metrics with Gemini
    # (This might vary slightly by Ragas version, but passing llm/embeddings to evaluate is standard)
    
    print("📊 Calculating Metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    
    print("\n📈 Evaluation Results:")
    print(results)
    
    # Save to CSV
    df = results.to_pandas()
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\n💾 Results saved to rag_evaluation_results.csv")

if __name__ == "__main__":
    main()
