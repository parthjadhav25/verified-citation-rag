from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from parser import load_pdfs

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks):
    print("Building search index...")
    texts = [chunk["text"] for chunk in chunks]
    
    # semantic index
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # bm25 index
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    
    print(f"Index built — {len(embeddings)} embeddings")
    return embeddings, bm25

def search(query, chunks, embeddings, bm25, top_k=3):
    # semantic search
    query_embedding = model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # bm25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # normalize both scores to 0-1
    semantic_scores = semantic_scores / (semantic_scores.max() + 1e-9)
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)
    
    # combine scores 50/50
    hybrid_scores = 0.5 * semantic_scores + 0.5 * bm25_scores
    
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": round(float(hybrid_scores[idx]), 3)
        })
    return results

if __name__ == "__main__":
    chunks = load_pdfs(".")
    embeddings, bm25 = build_index(chunks)
    
    query = "When was Doom II released?"
    print(f"\nQuery: {query}")
    results = search(query, chunks, embeddings, bm25)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {result['source']} — Page {result['page']}")
        print(f"Score: {result['score']}")
        print(f"Text preview: {result['text'][:200]}")