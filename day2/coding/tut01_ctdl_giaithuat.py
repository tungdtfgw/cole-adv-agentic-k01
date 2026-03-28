"""
=============================================================================
BÀI MẪU 1: Ứng dụng học Cấu trúc dữ liệu & Giải thuật
=============================================================================
Công nghệ: Python + Streamlit + Gemini Code Execution
Mô tả:
  - Người dùng chọn một CTDL hoặc thuật toán
  - Gemini giải thích ngắn gọn + viết code Python minh họa
  - Code được chạy trong sandbox → hiển thị code + kết quả

Cài đặt:
  pip install google-genai streamlit

Chạy:
  export GEMINI_API_KEY="your-api-key"
  streamlit run demo1_ctdl_giaithuat.py
=============================================================================
"""

import streamlit as st
from google import genai
from google.genai import types

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key


# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Học CTDL & Giải thuật với AI",
    page_icon="🧠",
    layout="wide",
)

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"], index=0)
    st.divider()
    st.markdown("""
    **Hướng dẫn:**
    1. Chọn chủ đề CTDL/Giải thuật
    2. Nhấn "Giải thích & Chạy code"
    3. Xem giải thích + code + kết quả
    """)

# ===== TIÊU ĐỀ =====
st.title("🧠 Học Cấu trúc dữ liệu & Giải thuật với AI")
st.markdown("Chọn một chủ đề → AI sẽ **giải thích** + **viết code minh họa** + **chạy code** trực tiếp!")

st.divider()

# ===== DANH SÁCH CHỦ ĐỀ =====
TOPICS = {
    "📦 Cấu trúc dữ liệu": {
        "Stack (Ngăn xếp)": "Cài đặt cấu trúc dữ liệu Stack bằng list trong Python. Minh họa các thao tác push, pop, peek kèm ví dụ. In ra trạng thái stack sau mỗi thao tác.",
        "Queue (Hàng đợi)": "Cài đặt cấu trúc dữ liệu Queue bằng collections.deque. Minh họa các thao tác enqueue, dequeue kèm ví dụ. In ra trạng thái queue sau mỗi thao tác.",
        "Linked List (Danh sách liên kết)": "Cài đặt Danh sách liên kết đơn với các thao tác thêm, xóa và duyệt. Tạo danh sách gồm 5 phần tử, xóa một phần tử, rồi in kết quả.",
        "Cây tìm kiếm nhị phân (BST)": "Cài đặt Cây tìm kiếm nhị phân với thao tác thêm node và duyệt inorder. Thêm các giá trị [50, 30, 70, 20, 40, 60, 80] rồi in kết quả duyệt inorder.",
        "Bảng băm (Hash Table)": "Cài đặt Bảng băm đơn giản với phương thức put và get, sử dụng chaining để xử lý đụng độ. Minh họa với 5 cặp key-value.",
        "Đồ thị (Graph)": "Cài đặt Đồ thị vô hướng bằng danh sách kề. Thêm các cạnh và thực hiện duyệt BFS từ một đỉnh xuất phát. In thứ tự duyệt.",
    },
    "⚡ Thuật toán sắp xếp": {
        "Sắp xếp nổi bọt (Bubble Sort)": "Cài đặt thuật toán Bubble Sort. Sắp xếp mảng [64, 34, 25, 12, 22, 11, 90]. In mảng sau mỗi lượt để thấy cách hoạt động từng bước.",
        "Sắp xếp nhanh (Quick Sort)": "Cài đặt thuật toán Quick Sort với hàm partition. Sắp xếp mảng [10, 80, 30, 90, 40, 50, 70]. In ra pivot được chọn và các phân hoạch tương ứng.",
        "Sắp xếp trộn (Merge Sort)": "Cài đặt thuật toán Merge Sort. Sắp xếp mảng [38, 27, 43, 3, 9, 82, 10]. Hiển thị các bước chia và trộn bằng cách in các mảng trung gian.",
        "Sắp xếp chèn (Insertion Sort)": "Cài đặt thuật toán Insertion Sort. Sắp xếp mảng [12, 11, 13, 5, 6]. In mảng sau mỗi bước chèn.",
    },
    "🔍 Thuật toán tìm kiếm": {
        "Tìm kiếm nhị phân (Binary Search)": "Cài đặt Tìm kiếm nhị phân (cả lặp và đệ quy). Tìm giá trị 23 và 50 trong mảng đã sắp xếp [2, 3, 4, 10, 23, 40, 67]. In từng bước so sánh.",
        "Tìm kiếm theo chiều sâu (DFS)": "Cài đặt DFS trên đồ thị bằng danh sách kề. Tạo đồ thị gồm 7 đỉnh, thực hiện DFS từ đỉnh 0. In thứ tự các đỉnh đã thăm.",
        "Tìm kiếm theo chiều rộng (BFS)": "Cài đặt BFS trên đồ thị bằng danh sách kề. Tạo đồ thị gồm 7 đỉnh, thực hiện BFS từ đỉnh 0. In thứ tự các đỉnh đã thăm.",
    },
    "🧩 Thuật toán khác": {
        "Fibonacci (Quy hoạch động)": "Tính dãy Fibonacci bằng 3 cách: đệ quy, đệ quy có nhớ (memoization), và quy hoạch động bottom-up. So sánh kết quả với n=10 và trình bày từng bước cho cách quy hoạch động.",
        "Dijkstra (Đường đi ngắn nhất)": "Cài đặt thuật toán Dijkstra tìm đường đi ngắn nhất. Tạo đồ thị có trọng số gồm 6 đỉnh, tìm đường đi ngắn nhất từ đỉnh 0 đến tất cả các đỉnh còn lại. In bảng khoảng cách.",
        "Kỹ thuật hai con trỏ (Two Pointers)": "Minh họa kỹ thuật Hai con trỏ: tìm hai số trong mảng đã sắp xếp [1, 2, 3, 4, 6, 8, 9, 14, 15] có tổng bằng 13. Hiển thị từng bước di chuyển con trỏ.",
    },
}

# ===== CHỌN CHỦ ĐỀ =====
col1, col2 = st.columns([1, 1])

with col1:
    category = st.selectbox("📂 Chọn nhóm", list(TOPICS.keys()))

with col2:
    topic = st.selectbox("📌 Chọn chủ đề", list(TOPICS[category].keys()))

# Cho phép người dùng tùy chỉnh prompt
custom_prompt = st.text_area(
    "✏️ Tùy chỉnh yêu cầu (không bắt buộc)",
    placeholder="Ví dụ: Hãy giải thích bằng tiếng Việt, thêm ví dụ với mảng khác...",
    height=68,
)

# ===== NÚT CHẠY =====
if st.button("🚀 Giải thích & Chạy code", type="primary", use_container_width=True):

    # Tạo prompt
    base_prompt = TOPICS[category][topic]
    prompt = f"""Bạn là một giảng viên Khoa học Máy tính. Hãy giải thích chủ đề sau một cách rõ ràng, dễ hiểu bằng tiếng Việt có dấu, sau đó viết code Python minh họa.

CHỦ ĐỀ: {topic}
YÊU CẦU: {base_prompt}

QUY TẮC:
- Giải thích khái niệm bằng tiếng Việt có dấu (2-3 đoạn, rõ ràng và dễ hiểu)
- Viết code Python sạch, có comment bằng tiếng Việt
- Code PHẢI in ra kết quả để thấy rõ output
- Thêm lệnh print để hiển thị từng bước thực thi
"""
    if custom_prompt:
        prompt += f"\n\nYÊU CẦU BỔ SUNG TỪ NGƯỜI DÙNG: {custom_prompt}"

    # Gọi Gemini API với Code Execution
    try:
        client = genai.Client(api_key=get_key())

        with st.spinner(f"🤖 AI đang phân tích **{topic}**..."):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                ),
            )

        # ===== HIỂN THỊ KẾT QUẢ =====
        st.divider()
        st.subheader(f"📘 {topic}")

        # Tách phần giải thích, code, và kết quả
        explanation_parts = []
        code_parts = []
        output_parts = []

        for part in response.candidates[0].content.parts:
            if part.text is not None and part.text.strip():
                explanation_parts.append(part.text)
            if part.executable_code is not None:
                code_parts.append(part.executable_code.code)
            if part.code_execution_result is not None:
                output_parts.append(part.code_execution_result.output)

        # Hiển thị giải thích
        for text in explanation_parts:
            st.markdown(text)

        # Hiển thị code và kết quả chạy song song
        if code_parts or output_parts:
            col_code, col_output = st.columns(2)
            with col_code:
                st.markdown("**💻 Code minh họa:**")
                for code in code_parts:
                    st.code(code, language="python")
            with col_output:
                st.markdown("**📊 Kết quả chạy trong sandbox:**")
                for output in output_parts:
                    st.code(output, language="text")

    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        st.info("💡 Kiểm tra lại API Key hoặc thử model khác.")

# ===== FOOTER =====
st.divider()
st.caption("💡 Ứng dụng sử dụng Gemini Code Execution — model tự viết code Python và chạy trong sandbox an toàn.")
