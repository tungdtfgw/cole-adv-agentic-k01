# SimpleNLM - Mô phỏng NotebookLM đơn giản

## Tổng quan

SimpleNLM là ứng dụng hỏi đáp tài liệu PDF sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)**. Người dùng upload PDF, hệ thống xử lý và lưu trữ nội dung dưới dạng vector, sau đó cho phép chat hỏi đáp với AI — câu trả lời luôn dựa trên nội dung tài liệu và có trích dẫn nguồn cụ thể.

## Kiến trúc hệ thống

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────┐
│  Upload PDF │────▶│  pdfplumber  │────▶│  Chunking   │────▶│fastembed│
│ (Streamlit) │     │ (trích text) │     │(chia đoạn)  │     │(embedding)
└─────────────┘     └──────────────┘     └─────────────┘     └────┬────┘
                                                                  │
                                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────┐
│  Hiển thị   │◀────│   Gemini     │◀────│  Prompt +   │◀────│  FAISS  │
│  (popover)  │     │ (sinh trả lời)│    │  Context    │     │ (search)│
└─────────────┘     └──────────────┘     └─────────────┘     └─────────┘
```

**Luồng xử lý gồm 2 pha:**

| Pha | Mô tả |
|-----|-------|
| **Indexing** (upload) | PDF → trích text → chia chunks → embedding → lưu vào FAISS index + disk |
| **Querying** (chat) | Câu hỏi → embedding → FAISS tìm top-K chunks → gửi Gemini kèm context → hiển thị kết quả với popover nguồn |

## Lựa chọn kỹ thuật

| Thành phần | Thư viện | Lý do chọn |
|------------|----------|-------------|
| **Trích xuất PDF** | `pdfplumber` | Trích text chính xác theo từng trang, giữ được metadata số trang để trích dẫn nguồn |
| **Embedding** | `fastembed` với model `paraphrase-multilingual-MiniLM-L12-v2` | Chạy local bằng ONNX Runtime (không cần GPU, không cần gọi API), hỗ trợ đa ngôn ngữ — cho phép hỏi tiếng Việt trên tài liệu tiếng Anh |
| **Vector search** | `FAISS` (IndexFlatIP) | Thư viện của Meta, tìm kiếm vector nhanh, hỗ trợ cosine similarity qua inner product + normalize |
| **Sinh câu trả lời** | `Gemini 3.0 Flash` | Nhanh, rẻ, hỗ trợ tiếng Việt tốt, context window lớn để nhận nhiều chunks cùng lúc |
| **Giao diện** | `Streamlit` | Xây dựng UI nhanh với chat interface, sidebar, popover có sẵn |

## Giải thích các đoạn code quan trọng

### 1. Tạo Chunks (Sliding Window)

```python
# simple_nlm.py dòng 43-53
def chunk_pages(pages):
    chunks = []
    for page_num, text in pages:
        words = text.split()
        for start in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[start : start + CHUNK_SIZE]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunks.append({"text": chunk_text, "page": page_num})
    return chunks
```

**Cách hoạt động:**
- Mỗi trang PDF được tách thành các đoạn (chunk) dài tối đa **500 từ**
- Cửa sổ trượt (sliding window) di chuyển với bước nhảy `500 - 100 = 400` từ
- **Overlap 100 từ** giữa 2 chunk liền kề đảm bảo nội dung nằm ở ranh giới không bị cắt đứt ngữ nghĩa
- Mỗi chunk giữ metadata `page` (số trang gốc) để trích dẫn sau này

**Ví dụ minh họa** với 1 trang có 1000 từ:
```
Chunk 1: từ 0   → 499   (500 từ)
Chunk 2: từ 400 → 899   (500 từ, overlap 100 từ với chunk 1)
Chunk 3: từ 800 → 999   (200 từ, overlap 100 từ với chunk 2)
```

### 2. Xây dựng FAISS Index

```python
# simple_nlm.py dòng 56-67
def build_index(chunks, embed_model):
    texts = [c["text"] for c in chunks]
    embeddings = list(embed_model.embed(texts))
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    return index, embeddings_np
```

**Cách hoạt động:**
- `embed_model.embed(texts)` biến mỗi chunk text thành vector 384 chiều
- `faiss.normalize_L2()` chuẩn hóa vector về độ dài 1 → inner product trở thành **cosine similarity**
- `IndexFlatIP` (Flat Inner Product) là brute-force search — so sánh với mọi vector, đảm bảo kết quả chính xác 100%
- Tất cả chunks từ **mọi file** được gộp vào **1 index duy nhất**, cho phép tìm kiếm xuyên tài liệu

### 3. Truy vấn (Semantic Search)

```python
# simple_nlm.py dòng 95-108
def search(query, index, chunks, embed_model, top_k=TOP_K):
    query_embedding = np.array(list(embed_model.embed([query]))).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "file": chunks[idx]["file"],
                "score": float(dist)
            })
    return results
```

**Cách hoạt động:**
- Câu hỏi của người dùng được embed thành vector cùng không gian với các chunks
- FAISS tìm **TOP_K = 5** chunks có cosine similarity cao nhất
- Kết quả trả về gồm: nội dung chunk, số trang, tên file, điểm tương đồng (score)
- Nhờ dùng model đa ngôn ngữ, câu hỏi tiếng Việt vẫn tìm được nội dung tiếng Anh tương ứng

### 4. Sinh câu trả lời với Gemini

```python
# simple_nlm.py dòng 111-146
def ask_gemini(question, context_chunks, chat_history):
    client = genai.Client(api_key=get_key())

    context = "\n\n".join(
        [f"[{c['file']} - Trang {c['page']}]\n{c['text']}" for c in context_chunks]
    )

    system_instruction = """...(quy tắc trả lời)..."""

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
        model=MODEL, contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text
```

**Cách hoạt động:**
- 5 chunks liên quan nhất được nối lại thành `context`, mỗi chunk gắn nhãn `[tên_file - Trang X]`
- **System instruction** yêu cầu Gemini:
  - Chỉ trả lời dựa trên tài liệu (không bịa)
  - Luôn ghi `[tên_file - Trang X]` sau mỗi thông tin trích dẫn
  - Thông báo rõ nếu tài liệu không có thông tin liên quan
- **Chat history** được truyền vào để Gemini hiểu ngữ cảnh các câu hỏi trước đó (hỗ trợ follow-up questions)

### 5. Tham chiếu tài liệu (Popover)

```python
# simple_nlm.py dòng 152-173
def render_answer_with_popovers(answer, sources):
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
                    st.caption(f"📄 **{matched_source['file']}** ...")
                    st.text(matched_source["text"][:500] + "...")
            else:
                st.markdown(f"**[{part}]**")
```

**Cách hoạt động:**
- Regex `\[...\]` tìm tất cả trích dẫn dạng `[file.pdf - Trang X]` trong câu trả lời của Gemini
- `re.split()` tách câu trả lời thành: text thường (index chẵn) và citation (index lẻ)
- Mỗi citation được thay bằng `st.popover` — một nút bấm mở popup hiển thị nội dung chunk gốc
- `source_map` ánh xạ citation text → chunk data để lấy đúng nội dung hiển thị

## Persistence (Lưu trữ)

Dữ liệu được lưu tự động vào thư mục `nlm_data/`:

| File | Nội dung |
|------|----------|
| `chunks.json` | Toàn bộ chunks (text + page + file) dạng JSON |
| `index.faiss` | FAISS index dạng binary |

Khi khởi động lại server, hệ thống tự động load từ disk — không cần upload lại PDF.

## Chạy ứng dụng

```bash
cd day4
streamlit run simple_nlm.py
```
