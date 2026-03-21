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
        "Stack (Ngăn xếp)": "Implement a Stack data structure using a Python list. Show push, pop, peek operations with examples. Print the stack after each operation.",
        "Queue (Hàng đợi)": "Implement a Queue data structure using collections.deque. Show enqueue, dequeue operations with examples. Print the queue after each operation.",
        "Linked List (Danh sách liên kết)": "Implement a Singly Linked List with insert, delete, and traverse operations. Create a list with 5 elements, delete one, and print the result.",
        "Binary Search Tree": "Implement a Binary Search Tree with insert and inorder traversal. Insert values [50, 30, 70, 20, 40, 60, 80] and print inorder result.",
        "Hash Table": "Implement a simple Hash Table with put and get methods using chaining for collision handling. Demonstrate with 5 key-value pairs.",
        "Graph (Đồ thị)": "Implement an undirected Graph using adjacency list. Add edges and perform BFS traversal from a starting node. Print the traversal order.",
    },
    "⚡ Thuật toán sắp xếp": {
        "Bubble Sort": "Implement Bubble Sort. Sort the array [64, 34, 25, 12, 22, 11, 90]. Print the array after each pass to show how it works step by step.",
        "Quick Sort": "Implement Quick Sort with partition function. Sort the array [10, 80, 30, 90, 40, 50, 70]. Print the pivot choices and resulting partitions.",
        "Merge Sort": "Implement Merge Sort. Sort the array [38, 27, 43, 3, 9, 82, 10]. Show the divide and merge steps by printing intermediate arrays.",
        "Insertion Sort": "Implement Insertion Sort. Sort the array [12, 11, 13, 5, 6]. Print the array after each insertion step.",
    },
    "🔍 Thuật toán tìm kiếm": {
        "Binary Search": "Implement Binary Search (iterative and recursive). Search for values 23 and 50 in sorted array [2, 3, 4, 10, 23, 40, 67]. Print each comparison step.",
        "Depth-First Search (DFS)": "Implement DFS on a graph with adjacency list. Create a graph with 7 nodes, perform DFS from node 0. Print visited nodes in order.",
        "Breadth-First Search (BFS)": "Implement BFS on a graph with adjacency list. Create a graph with 7 nodes, perform BFS from node 0. Print visited nodes in order.",
    },
    "🧩 Thuật toán khác": {
        "Fibonacci (DP)": "Calculate Fibonacci numbers using 3 approaches: recursive, memoization, and bottom-up DP. Compare the results for n=10 and show step-by-step for DP approach.",
        "Dijkstra (Đường đi ngắn nhất)": "Implement Dijkstra's shortest path algorithm. Create a weighted graph with 6 nodes and find shortest paths from node 0 to all others. Print the distance table.",
        "Two Pointers": "Demonstrate Two Pointers technique: find two numbers in sorted array [1, 2, 3, 4, 6, 8, 9, 14, 15] that sum to 13. Show each pointer movement.",
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
    prompt = f"""You are a computer science teacher. Explain the following topic clearly and concisely in Vietnamese, then write Python code to demonstrate it.

TOPIC: {topic}
TASK: {base_prompt}

REQUIREMENTS:
- Explain the concept in Vietnamese (2-3 paragraphs, clear and easy to understand)
- Write clean, well-commented Python code
- The code MUST print output to show results clearly
- Add print statements to show step-by-step execution
"""
    if custom_prompt:
        prompt += f"\n\nADDITIONAL REQUEST FROM USER: {custom_prompt}"

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
