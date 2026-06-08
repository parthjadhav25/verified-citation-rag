import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

def generate_answer(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}, Page {chunk['page']}]\n"
        context += chunk["text"][:500]
        context += "\n"
    
    prompt = f"""You are a helpful assistant that answers questions based only on the provided sources.

For every fact you state, you MUST cite the source like this: [Source 1], [Source 2], etc.
If the answer is not in the sources, say "I cannot find this in the provided documents."
Never make up information.

Sources:
{context}

Question: {query}

Answer with citations:"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    
    data = response.json()

    print(data)
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    test_chunks = [
        {
            "text": "Hotline Miami is a top-down shooter developed by Dennaton Games, consisting of Jonatan Söderström and Dennis Wedin. It was published by Devolver Digital.",
            "source": "Hotline Miami - Wikipedia.pdf",
            "page": 1
        }
    ]
    
    answer = generate_answer("Who developed Hotline Miami?", test_chunks)
    print(answer)