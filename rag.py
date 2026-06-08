from parser import load_pdfs
from search import build_index, search
from generator import generate_answer

def run_rag(query, folder_path="."):
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    chunks = load_pdfs(folder_path)
    embeddings, bm25 = build_index(chunks)
    relevant_chunks = search(query, chunks, embeddings, bm25, top_k=3)
    
    print(f"\nTop {len(relevant_chunks)} chunks found:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"  {i+1}. {chunk['source']} — Page {chunk['page']} (score: {chunk['score']})")
    
    print("\nGenerating answer...")
    answer = generate_answer(query, relevant_chunks)
    
    print(f"\nAnswer:\n{answer}")
    return answer

if __name__ == "__main__":
    queries = [
        "Who developed Hotline Miami?",
        "What is the plot of Cry of Fear?",
        "When was Doom II released?"
    ]
    
    for query in queries:
        run_rag(query)
        print("\n" + "=" * 50)