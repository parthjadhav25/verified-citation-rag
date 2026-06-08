import streamlit as st
from parser import load_pdfs
from search import build_index, search
from generator import generate_answer

st.set_page_config(page_title="Verified Citation RAG", layout="wide")

st.markdown("""
    <style>
        .main { font-size: 18px; }
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 1.8rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎮 Verified Citation RAG")
st.subheader("Gaming Wikipedia — Ask anything, get cited answers")

@st.cache_resource
def load_everything():
    chunks = load_pdfs(".")
    embeddings, bm25 = build_index(chunks)
    return chunks, embeddings, bm25

with st.spinner("Loading documents and building search index..."):
    chunks, embeddings, bm25 = load_everything()

st.success(f"✅ Loaded {len(chunks)} chunks from 3 documents")

st.header("🔍 Ask a Question")
query = st.text_input("Type your question here:", placeholder="Who developed Hotline Miami?")

if query:
    with st.spinner("Searching and generating answer..."):
        relevant_chunks = search(query, chunks, embeddings, bm25, top_k=3)
        answer = generate_answer(query, relevant_chunks)

    st.header("💬 Answer")
    st.write(answer)

    st.header("📄 Source Chunks")
    for i, chunk in enumerate(relevant_chunks):
        with st.expander(f"Source {i+1} — {chunk['source']} — Page {chunk['page']} (score: {chunk['score']})"):
            st.write(chunk["text"])