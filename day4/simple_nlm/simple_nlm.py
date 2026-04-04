import sys
import os
import json
import streamlit as st
import numpy as np
import faiss
import pdfplumber
from fastembed import TextEmbedding

from google import genai
from google.genai import types

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import get_key

MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlm_data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")


@st.cache_resource
def load_embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file, returning list of (page_num, text)."""
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i + 1, text.strip()))
    return pages


def chunk_pages(pages):
    """Split pages into smaller chunks for better retrieval."""
    chunks = []
    for page_num, text in pages:
        words = text.split()
        for start in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[start : start + CHUNK_SIZE]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunks.append({"text": chunk_text, "page": page_num})
    return chunks


def build_index(chunks, embed_model):
    """Build FAISS index from text chunks."""
    texts = [c["text"] for c in chunks]
    embeddings = list(embed_model.embed(texts))
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    return index, embeddings_np


def save_to_disk(chunks, index):
    """Save chunks and FAISS index to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    faiss.write_index(index, INDEX_FILE)


def load_from_disk():
    """Load chunks and FAISS index from disk. Returns (chunks, index) or (None, None)."""
    if not os.path.exists(CHUNKS_FILE) or not os.path.exists(INDEX_FILE):
        return None, None
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index(INDEX_FILE)
    return chunks, index


def clear_disk():
    """Remove saved data from disk."""
    for f in [CHUNKS_FILE, INDEX_FILE]:
        if os.path.exists(f):
            os.remove(f)


def search(query, index, chunks, embed_model, top_k=TOP_K):
    """Search for relevant chunks given a query."""
    query_embedding = np.array(list(embed_model.embed([query]))).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append(
                {"text": chunks[idx]["text"], "page": chunks[idx]["page"], "file": chunks[idx]["file"], "score": float(dist)}
            )
    return results


def ask_gemini(question, context_chunks, chat_history):
    """Send question with context to Gemini and get response."""
    client = genai.Client(api_key=get_key())

    context = "\n\n".join(
        [f"[{c['file']} - Trang {c['page']}]\n{c['text']}" for c in context_chunks]
    )

    system_instruction = """Bạn là trợ lý AI chuyên phân tích tài liệu. Hãy trả lời câu hỏi dựa trên nội dung tài liệu được cung cấp.

Quy tắc:
1. Chỉ trả lời dựa trên thông tin có trong tài liệu.
2. Luôn trích dẫn nguồn bằng cách ghi [tên_file - Trang X] sau mỗi thông tin, ví dụ: [report.pdf - Trang 3].
3. Nếu thông tin không có trong tài liệu, hãy nói rõ "Tài liệu không đề cập đến vấn đề này".
4. Trả lời bằng tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh."""

    messages = []
    for msg in chat_history:
        messages.append(types.Content(role=msg["role"], parts=[types.Part(text=msg["text"])]))

    user_prompt = f"""Nội dung tài liệu liên quan:
---
{context}
---

Câu hỏi: {question}"""

    messages.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))

    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )

    return response.text


import re


def render_answer_with_popovers(answer, sources):
    """Render answer text, replacing [file - Trang X] with clickable popovers."""
    source_map = {}
    for s in sources:
        key = f"{s['file']} - Trang {s['page']}"
        source_map[key] = s

    pattern = r'\[([^\]]+?\.pdf\s*-\s*Trang\s*\d+)\]'
    parts = re.split(pattern, answer)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            st.markdown(part, unsafe_allow_html=False)
        else:
            normalized = re.sub(r'\s+', ' ', part.strip())
            matched_source = source_map.get(normalized)
            if matched_source:
                with st.popover(f"📎 {normalized}"):
                    st.caption(f"📄 **{matched_source['file']}** - Trang {matched_source['page']}")
                    st.text(matched_source["text"][:500] + "..." if len(matched_source["text"]) > 500 else matched_source["text"])
            else:
                st.markdown(f"**[{part}]**")


# ============ Streamlit UI ============

st.set_page_config(page_title="SimpleNLM", page_icon="📚", layout="wide")
st.title("📚 SimpleNLM - NotebookLM đơn giản")
st.caption("Upload tài liệu PDF và chat hỏi đáp với AI")

embed_model = load_embedding_model()

# --- Session state (auto-load from disk) ---
if "documents" not in st.session_state:
    saved_chunks, saved_index = load_from_disk()
    if saved_chunks and saved_index:
        # Rebuild documents dict from saved chunks
        documents = {}
        for c in saved_chunks:
            documents.setdefault(c["file"], []).append(c)
        st.session_state.documents = documents
        st.session_state.all_chunks = saved_chunks
        st.session_state.index = saved_index
    else:
        st.session_state.documents = {}
        st.session_state.all_chunks = []
        st.session_state.index = None
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # for Gemini context

# --- Sidebar: Upload PDF ---
with st.sidebar:
    st.header("📄 Tài liệu")
    uploaded_files = st.file_uploader(
        "Upload PDF", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        new_files = []
        for f in uploaded_files:
            if f.name not in st.session_state.documents:
                new_files.append(f)

        if new_files:
            with st.spinner("Đang xử lý tài liệu..."):
                for f in new_files:
                    pages = extract_text_from_pdf(f)
                    chunks = chunk_pages(pages)
                    # Tag chunks with filename
                    for c in chunks:
                        c["file"] = f.name
                    st.session_state.documents[f.name] = chunks

                # Rebuild index with all chunks
                all_chunks = []
                for chunks in st.session_state.documents.values():
                    all_chunks.extend(chunks)
                st.session_state.all_chunks = all_chunks

                if all_chunks:
                    index, _ = build_index(all_chunks, embed_model)
                    st.session_state.index = index
                    save_to_disk(all_chunks, index)

            st.success(f"Đã xử lý {len(new_files)} tài liệu mới!")

    # Show uploaded documents
    if st.session_state.documents:
        st.divider()
        st.subheader("Tài liệu đã upload")
        for name, chunks in st.session_state.documents.items():
            st.write(f"📄 **{name}** — {len(chunks)} chunks")

    # Clear button
    if st.session_state.documents:
        if st.button("🗑️ Xóa tất cả", use_container_width=True):
            st.session_state.documents = {}
            st.session_state.index = None
            st.session_state.all_chunks = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            clear_disk()
            st.rerun()

# --- Main: Chat ---
if not st.session_state.documents:
    st.info("👈 Hãy upload tài liệu PDF ở sidebar để bắt đầu.")
else:
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "sources" in msg:
                render_answer_with_popovers(msg["content"], msg["sources"])
            else:
                st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Hỏi về nội dung tài liệu..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Search & answer
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và phân tích..."):
                results = search(
                    question,
                    st.session_state.index,
                    st.session_state.all_chunks,
                    embed_model,
                )

                answer = ask_gemini(question, results, st.session_state.chat_history)

            render_answer_with_popovers(answer, results)

        # Save to history
        sources = [{"file": r["file"], "page": r["page"], "score": r["score"], "text": r["text"]} for r in results]
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        # Keep chat history for Gemini (without context, just Q&A)
        st.session_state.chat_history.append({"role": "user", "text": question})
        st.session_state.chat_history.append({"role": "model", "text": answer})
