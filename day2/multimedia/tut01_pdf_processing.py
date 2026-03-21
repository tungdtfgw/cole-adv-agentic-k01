"""
Demo 1: Ứng dụng xử lý tài liệu PDF với Gemini API
- Bài 1: Upload PDF → Tóm tắt nội dung
- Bài 2: Upload PDF → Trích xuất bảng biểu ra Markdown

Yêu cầu:
    pip install google-genai streamlit

Chạy:
    export GEMINI_API_KEY="your-api-key"
    streamlit run demo1_pdf_processing.py
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
    page_title="Gemini PDF Processor",
    page_icon="📄",
    layout="wide",
)

MODEL = "gemini-3-flash-preview"


# ============ INIT CLIENT ============
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())  # Reads GEMINI_API_KEY from env


# ============ HELPER: Upload PDF to Gemini Files API ============
def upload_pdf(uploaded_file):
    """Upload a Streamlit UploadedFile to Gemini Files API."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    client = get_client()
    gemini_file = client.files.upload(
        file=pathlib.Path(tmp_path),
        config=dict(mime_type="application/pdf"),
    )
    return gemini_file


# ============ TASK 1: Summarize PDF ============
def summarize_pdf(gemini_file, language="Vietnamese"):
    client = get_client()

    prompt = f"""Hãy tóm tắt tài liệu PDF này một cách chi tiết và có cấu trúc.

Yêu cầu:
1. Tóm tắt tổng quan (2-3 câu) về nội dung chính của tài liệu.
2. Liệt kê các điểm chính / chương / phần quan trọng.
3. Với mỗi phần, tóm tắt nội dung cốt lõi trong 1-2 câu.
4. Nêu kết luận hoặc điểm đáng chú ý nhất.

Trả lời bằng tiếng {language}. Sử dụng Markdown formatting."""

    response = client.models.generate_content(
        model=MODEL,
        contents=[gemini_file, prompt],
    )
    return response.text


# ============ TASK 2: Extract Tables to Markdown ============
def extract_tables(gemini_file):
    client = get_client()

    prompt = """Hãy trích xuất TẤT CẢ các bảng biểu có trong tài liệu PDF này.

Yêu cầu:
1. Với mỗi bảng, ghi rõ tiêu đề hoặc ngữ cảnh của bảng (nằm ở trang nào, phần nào).
2. Chuyển đổi bảng sang định dạng Markdown table.
3. Giữ nguyên dữ liệu gốc, không thêm hoặc bớt thông tin.
4. Nếu bảng có ghi chú hoặc footnote, ghi kèm bên dưới.
5. Nếu không tìm thấy bảng nào, hãy thông báo rõ ràng.

Trả lời bằng Markdown."""

    response = client.models.generate_content(
        model=MODEL,
        contents=[gemini_file, prompt],
    )
    return response.text


# ============ UI ============
st.title("📄 Gemini PDF Processor")
st.caption("Tóm tắt tài liệu & Trích xuất bảng biểu từ PDF với Gemini API")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    task = st.radio(
        "Chọn tác vụ:",
        ["📝 Tóm tắt tài liệu", "📊 Trích xuất bảng biểu"],
        index=0,
    )
    language = st.selectbox(
        "Ngôn ngữ đầu ra:",
        ["Vietnamese", "English"],
        index=0,
    )
    st.divider()
    st.markdown(
        """
    **Hướng dẫn:**
    1. Upload file PDF ở vùng bên phải
    2. Chọn tác vụ muốn thực hiện
    3. Nhấn nút xử lý
    
    **Giới hạn:**
    - File tối đa: 50 MB
    - Tối đa: 1000 trang
    """
    )

# Main area
uploaded_file = st.file_uploader(
    "Upload file PDF",
    type=["pdf"],
    help="Chọn file PDF cần xử lý (tối đa 50MB, 1000 trang)",
)

if uploaded_file is not None:
    # Show file info
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"📎 **{uploaded_file.name}** — {file_size_mb:.1f} MB")

    col1, col2 = st.columns([1, 3])

    with col1:
        process_btn = st.button("🚀 Xử lý", type="primary", use_container_width=True)

    if process_btn:
        # Upload to Gemini
        with st.spinner("Đang upload PDF lên Gemini Files API..."):
            gemini_file = upload_pdf(uploaded_file)

        # Process
        if task == "📝 Tóm tắt tài liệu":
            with st.spinner("Gemini đang tóm tắt tài liệu..."):
                result = summarize_pdf(gemini_file, language)

            st.subheader("📝 Kết quả tóm tắt")
            st.markdown(result)

        else:  # Extract tables
            with st.spinner("Gemini đang trích xuất bảng biểu..."):
                result = extract_tables(gemini_file)

            st.subheader("📊 Bảng biểu trích xuất")
            st.markdown(result)

        # Download result
        st.divider()
        st.download_button(
            label="💾 Tải kết quả (Markdown)",
            data=result,
            file_name=f"{uploaded_file.name}_result.md",
            mime="text/markdown",
        )
else:
    # Empty state
    st.markdown(
        """
    <div style="text-align: center; padding: 60px 20px; color: #888;">
        <h3>👆 Upload một file PDF để bắt đầu</h3>
        <p>Hỗ trợ tóm tắt tài liệu và trích xuất bảng biểu</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
