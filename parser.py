import fitz
import os

def load_pdfs(folder_path):
    chunks = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            doc = fitz.open(filepath)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if len(text.strip()) > 50:
                    chunks.append({
                        "text": text,
                        "source": filename,
                        "page": page_num + 1
                    })
            
            print(f"Loaded {filename} — {len(doc)} pages")
    
    return chunks

if __name__ == "__main__":
    chunks = load_pdfs(".")
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\nSample chunk:")
    print(f"Source: {chunks[0]['source']}")
    print(f"Page: {chunks[0]['page']}")
    print(f"Text preview: {chunks[0]['text'][:200]}")