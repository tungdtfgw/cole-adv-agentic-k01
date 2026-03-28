"""
=============================================================================
BÀI MẪU 2: Tool phân tích dữ liệu tự động
=============================================================================
Công nghệ: Python + Streamlit + Gemini Code Execution
Mô tả:
  - Người dùng upload CSV/Excel (mặc định dùng penguins_simple.csv)
  - Gemini tự viết code phân tích, vẽ chart, trả insight
  - Hiển thị code + kết quả + biểu đồ trên giao diện Streamlit

Cài đặt:
  pip install google-genai streamlit pandas

Chạy:
  export GEMINI_API_KEY="your-api-key"
  streamlit run demo2_data_analysis.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import base64
import io
from google import genai
from google.genai import types

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"], index=0)
    st.divider()
    st.markdown("""
    **Cách dùng:**
    1. Upload file CSV/Excel (hoặc dùng file mẫu)
    2. Chọn kiểu phân tích
    3. Xem kết quả: code + insight + biểu đồ
    """)

# ===== TIÊU ĐỀ =====
st.title("📊 AI Data Analyst")
st.markdown("Upload dữ liệu → AI tự viết code phân tích, vẽ biểu đồ, và trả insight!")

st.divider()

# ===== UPLOAD FILE =====
uploaded_file = st.file_uploader(
    "📁 Upload file CSV hoặc Excel",
    type=["csv", "xlsx", "xls", "tsv"],
    help="Hỗ trợ: CSV, TSV, Excel (.xlsx, .xls)"
)

# Nút dùng file mẫu
use_sample = st.checkbox("📌 Hoặc dùng file mẫu: penguins_simple.csv")

# ===== ĐỌC DỮ LIỆU =====
df = None
file_content_str = None
filename = None

if uploaded_file is not None:
    filename = uploaded_file.name
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            file_content_str = uploaded_file.read().decode("utf-8")
        elif filename.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep="\t")
            uploaded_file.seek(0)
            file_content_str = uploaded_file.read().decode("utf-8")
        else:
            df = pd.read_excel(uploaded_file)
            # Với Excel, convert sang CSV string để gửi cho Gemini
            file_content_str = df.to_csv(index=False)
            filename = filename.rsplit(".", 1)[0] + ".csv"
    except Exception as e:
        st.error(f"❌ Không đọc được file: {e}")

elif use_sample:
    filename = "penguins_simple.csv"
    try:
        df = pd.read_csv("penguins_simple.csv")
        file_content_str = df.to_csv(index=False)
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file penguins_simple.csv. Hãy đặt file cùng thư mục với script.")

# ===== HIỂN THỊ PREVIEW =====
if df is not None:
    st.subheader(f"👀 Preview: {filename}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng", f"{len(df):,}")
    col2.metric("Số cột", len(df.columns))
    col3.metric("Missing values", f"{df.isnull().sum().sum():,}")

    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # ===== CHỌN KIỂU PHÂN TÍCH =====
    analysis_options = {
        "🔍 Khám phá tổng quan (EDA)":
            "Perform a comprehensive EDA: show data shape, dtypes, missing values, "
            "basic statistics (mean, median, std, min, max for numeric columns), "
            "value counts for categorical columns. Create 2-3 matplotlib charts "
            "(histogram, bar chart, box plot) to visualize key patterns.",

        "📈 Phân tích tương quan":
            "Analyze correlations between all numeric columns. "
            "Create a correlation heatmap using matplotlib/seaborn. "
            "Identify the top 3 strongest correlations and explain what they mean.",

        "📊 Phân tích theo nhóm":
            "Identify the best categorical column for grouping. "
            "Calculate group-level statistics (mean, count, std) for numeric columns. "
            "Create bar charts comparing groups. Highlight key differences between groups.",

        "🧹 Phát hiện Outlier & Missing Data":
            "Analyze missing data patterns (which columns, how many, percentage). "
            "Detect outliers using IQR method for each numeric column. "
            "Create box plots to visualize outliers. Suggest data cleaning strategies.",

        "💬 Hỏi tùy chỉnh":
            None,  # Sẽ dùng text input
    }

    analysis_type = st.selectbox("📋 Chọn kiểu phân tích", list(analysis_options.keys()))

    # Custom question
    custom_question = ""
    if analysis_type == "💬 Hỏi tùy chỉnh":
        custom_question = st.text_area(
            "Nhập câu hỏi về dữ liệu",
            placeholder="Ví dụ: So sánh cân nặng trung bình giữa các loài penguin theo giới tính...",
            height=80,
        )

    # ===== NÚT PHÂN TÍCH =====
    if st.button("🚀 Phân tích ngay!", type="primary", use_container_width=True):
        if analysis_type == "💬 Hỏi tùy chỉnh" and not custom_question:
            st.error("⚠️ Vui lòng nhập câu hỏi!")
            st.stop()

        # Tạo prompt
        task_description = analysis_options[analysis_type] or custom_question
        # Chỉ gửi tối đa 50 dòng đầu + schema để tiết kiệm token
        sample_data = file_content_str if len(df) <= 100 else df.head(50).to_csv(index=False)

        prompt = f"""You are a senior data analyst. Analyze the following dataset and provide insights.

DATASET INFO:
- Filename: {filename}
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Columns: {list(df.columns)}
- Dtypes: {dict(df.dtypes.astype(str))}

DATA (CSV format):
```
{sample_data}
```

TASK: {task_description}

REQUIREMENTS:
- Write clean Python code using pandas, matplotlib
- Use Vietnamese for all explanations and chart labels/titles
- ALL charts MUST use matplotlib (the only supported library for rendering)
- Use plt.figure(figsize=...) for readable charts
- Always call plt.tight_layout() before plt.show()
- Print clear summary statistics with descriptive labels
- End with 3-5 KEY INSIGHTS in Vietnamese (bullet points)
"""

        # Gọi Gemini
        try:
            client = genai.Client(api_key=get_key())

            with st.spinner("🤖 AI đang phân tích dữ liệu..."):
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    ),
                )

            # ===== HIỂN THỊ KẾT QUẢ =====
            st.divider()
            st.subheader("📋 Kết quả phân tích")

            for part in response.candidates[0].content.parts:
                # Text giải thích / insight
                if part.text is not None and part.text.strip():
                    st.markdown(part.text)

                # Biểu đồ từ sandbox
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    st.image(part.inline_data.data, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
            st.info("💡 Thử model khác hoặc kiểm tra API Key.")

# ===== FOOTER =====
st.divider()
st.caption("💡 Sử dụng Gemini Code Execution — AI tự viết code Python phân tích dữ liệu trong sandbox an toàn.")
