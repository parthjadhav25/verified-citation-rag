from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_claims(answer):
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    claims = []
    for sentence in sentences:
        source_match = re.findall(r'\[Source (\d+)\]', sentence)
        if source_match:
            clean = re.sub(r'\[Source \d+\]', '', sentence).strip()
            claims.append({
                "claim": clean,
                "source_ids": [int(x) - 1 for x in source_match]
            })
    return claims

def verify_claims(claims, chunks):
    results = []
    for claim in claims:
        claim_embedding = model.encode([claim["claim"]])
        verified = False
        best_score = 0
        
        for source_id in claim["source_ids"]:
            if source_id < len(chunks):
                chunk_embedding = model.encode([chunks[source_id]["text"][:500]])
                score = cosine_similarity(claim_embedding, chunk_embedding)[0][0]
                if score > best_score:
                    best_score = score
                if score >= 0.3:
                    verified = True
        
        results.append({
            "claim": claim["claim"],
            "verified": verified,
            "confidence": round(float(best_score), 3)
        })
    return results

if __name__ == "__main__":
    test_answer = "Hotline Miami was developed by Dennaton Games [Source 1]. The game was released in 1994 [Source 1]."
    test_chunks = [
        {
            "text": "Hotline Miami is a top-down shooter developed by Dennaton Games, consisting of Jonatan Söderström and Dennis Wedin.",
            "source": "Hotline Miami - Wikipedia.pdf",
            "page": 1
        }
    ]
    
    claims = extract_claims(test_answer)
    results = verify_claims(claims, test_chunks)
    
    print("Verification results:")
    for r in results:
        status = "✅ VERIFIED" if r["verified"] else "❌ UNVERIFIED"
        print(f"{status} (confidence: {r['confidence']})")
        print(f"Claim: {r['claim']}")
        print()