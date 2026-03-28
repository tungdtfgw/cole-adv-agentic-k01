"""
Ví dụ 4: Hỏi đáp PDF với Context Caching (Streamlit)
- Lần đầu: upload PDF → tạo cache → hỏi đáp
- Lần sau: phát hiện cache còn sống → hỏi đáp luôn, không cần upload lại

Cài đặt:
  pip install google-genai streamlit python-dotenv

Chạy:
  streamlit run tut04_streamlit_cached_qa.py
"""
import streamlit as st
from google import genai
from google.genai import types
import tempfile
import pathlib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key

# ============ CONFIG ============
st.set_page_config(
    page_title="PDF Q&A with Caching",
    page_icon="📚",
    layout="wide",
)

MODEL = "gemini-3-flash-preview"
CACHE_DISPLAY_NAME = "pdf_qa_streamlit"
CACHE_TTL = "1800s"  # 30 phút

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thông minh. "
    "Trả lời câu hỏi dựa HOÀN TOÀN vào tài liệu PDF được cung cấp. "
    "Trả lời bằng tiếng Việt. "
    "Nếu không tìm thấy thông tin trong tài liệu, hãy nói rõ."
)


# ============ INIT CLIENT ============
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())


# ============ TÌM CACHE CÒN SỐNG ============
def find_existing_cache(client):
    """Tìm cache có display_name trùng khớp và còn hạn."""
    for c in client.caches.list():
        if c.display_name == CACHE_DISPLAY_NAME:
            return c
    return None


# ============ TẠO CACHE TỪ PDF ============
def create_cache_from_pdf(client, uploaded_file):
    """Upload PDF lên Gemini và tạo cache."""
    # Upload file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    gemini_file = client.files.upload(
        file=pathlib.Path(tmp_path),
        config=dict(mime_type="application/pdf"),
    )

    # Tạo cache
    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            display_name=CACHE_DISPLAY_NAME,
            system_instruction=SYSTEM_INSTRUCTION,
            contents=[gemini_file],
            ttl=CACHE_TTL,
        )
    )
    return cache


# ============ UI ============
st.title("📚 PDF Q&A with Context Caching")
st.caption("Hỏi đáp tài liệu PDF — cache giữ lại giữa các phiên, không cần upload lại")

client = get_client()

# --- Kiểm tra cache ---
cache = find_existing_cache(client)

# --- Sidebar: trạng thái cache ---
with st.sidebar:
    st.header("⚙️ Cache Status")

    if cache:
        st.success(f"Cache đang hoạt động")
        st.markdown(f"""
- **Name:** `{cache.name}`
- **Model:** `{cache.model}`
- **Hết hạn:** `{cache.expire_time}`
        """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Gia hạn", use_container_width=True):
                client.caches.update(
                    name=cache.name,
                    config=types.UpdateCachedContentConfig(ttl=CACHE_TTL)
                )
                st.success(f"Đã gia hạn thêm {CACHE_TTL}!")
                st.rerun()
        with col2:
            if st.button("🗑️ Xoá cache", use_container_width=True):
                client.caches.delete(cache.name)
                st.session_state.messages = []
                st.success("Đã xoá cache!")
                st.rerun()
    else:
        st.warning("Chưa có cache. Upload PDF để bắt đầu.")

    st.divider()
    st.markdown(f"""
**Cách hoạt động:**
1. Upload PDF → tạo cache (TTL {CACHE_TTL})
2. Hỏi đáp — mỗi câu chỉ gửi question, không gửi lại PDF
3. Đóng app, mở lại → cache vẫn còn, hỏi đáp tiếp
4. Hết TTL hoặc xoá thủ công → upload PDF mới
    """)

# --- Khu vực chính ---
if not cache:
    # === CHƯA CÓ CACHE: hiển thị upload ===
    st.markdown("### 📄 Upload PDF để tạo cache")
    uploaded_file = st.file_uploader(
        "Chọn file PDF",
        type=["pdf"],
        help="File PDF sẽ được upload lên Gemini và cache lại",
    )

    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📎 **{uploaded_file.name}** — {file_size_mb:.1f} MB")

        if st.button("🚀 Tạo cache & bắt đầu hỏi đáp", type="primary"):
            with st.spinner("Đang upload PDF và tạo cache..."):
                cache = create_cache_from_pdf(client, uploaded_file)
            st.success(f"Cache đã tạo: `{cache.name}`")
            st.session_state.messages = []
            st.rerun()
else:
    # === ĐÃ CÓ CACHE: giao diện chat ===
    st.markdown("### 💬 Hỏi đáp về nội dung PDF")

    # Khởi tạo chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "usage" in msg:
                u = msg["usage"]
                st.caption(
                    f"Input: {u['input']:,} | "
                    f"Cached: {u['cached']:,} | "
                    f"Output: {u['output']:,}"
                )

    # Input câu hỏi
    if question := st.chat_input("Hỏi về nội dung PDF..."):
        # Hiển thị câu hỏi
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Gửi request với cache
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                response = client.models.generate_content(
                    model=MODEL,
                    contents=question,
                    config=types.GenerateContentConfig(
                        cached_content=cache.name
                    )
                )

            answer = response.text
            st.markdown(answer)

            # Hiển thị token usage
            usage = response.usage_metadata
            usage_info = {
                "input": usage.prompt_token_count,
                "cached": getattr(usage, 'cached_content_token_count', 0) or 0,
                "output": usage.candidates_token_count,
            }
            st.caption(
                f"Input: {usage_info['input']:,} | "
                f"Cached: {usage_info['cached']:,} | "
                f"Output: {usage_info['output']:,}"
            )

        # Lưu vào history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "usage": usage_info,
        })
