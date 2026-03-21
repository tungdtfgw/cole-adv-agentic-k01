"""
=============================================================================
BÀI MẪU 4: Function Calling — Gọi Genderize API
=============================================================================
Công nghệ: Python + Streamlit + Gemini Function Calling
Mô tả:
  - Khai báo hàm predict_gender(name) gọi API https://api.genderize.io
  - Gemini tự gọi hàm qua Function Calling (hỗ trợ nhiều tên = Parallel FC)
  - Hiển thị kết quả dự đoán giới tính trên giao diện Streamlit

Cài đặt:
  pip install google-genai streamlit requests

Chạy:
  export GEMINI_API_KEY="your-api-key"
  streamlit run demo4_genderize.py
=============================================================================
"""

import streamlit as st
import requests
from google import genai
from google.genai import types

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key

# ===== CONFIG =====
st.set_page_config(page_title="Gender Predictor", page_icon="🧑")

MODEL = "gemini-2.5-flash"

# Function Declaration
predict_gender_declaration = {
    "name": "predict_gender",
    "description": "Predict gender (male/female) based on a person's name "
                   "using the Genderize.io API. Returns gender, probability, and sample count.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's name to predict gender for (e.g. 'Alex', 'Linh', 'James').",
            }
        },
        "required": ["name"],
    },
}


# Hàm thực tế gọi API
def predict_gender(name: str) -> dict:
    """Gọi Genderize API để dự đoán giới tính từ tên."""
    response = requests.get("https://api.genderize.io", params={"name": name}, timeout=10)
    response.raise_for_status()
    data = response.json()
    return {
        "name": data.get("name", name),
        "gender": data.get("gender", "unknown"),
        "probability": data.get("probability", 0),
        "count": data.get("count", 0),
    }


# Gemini client
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())


def predict_via_gemini(client, names_text):
    """Dùng Function Calling (hỗ trợ Parallel) để dự đoán giới tính."""
    tools = types.Tool(function_declarations=[predict_gender_declaration])
    config = types.GenerateContentConfig(
        tools=[tools],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    user_prompt = f"Dự đoán giới tính cho các tên sau: {names_text}. Trả lời bằng tiếng Việt."
    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    # Gửi prompt → Model trả về function_call(s)
    response = client.models.generate_content(
        model=MODEL, contents=contents, config=config,
    )

    function_calls = response.function_calls
    if not function_calls:
        return response.text

    # Chạy tất cả hàm
    contents.append(response.candidates[0].content)

    function_response_parts = []
    for fc in function_calls:
        result = predict_gender(**fc.args)
        function_response_parts.append(
            types.Part.from_function_response(name=fc.name, response={"result": result})
        )

    contents.append(types.Content(role="user", parts=function_response_parts))

    # Gửi kết quả lại → Nhận câu trả lời cuối
    final_response = client.models.generate_content(
        model=MODEL, contents=contents, config=config,
    )
    return final_response.text


# ===== UI =====
st.title("🧑 Gender Predictor")
st.caption("Function Calling demo — Gemini gọi Genderize API để dự đoán giới tính từ tên")

names_input = st.text_input(
    "Nhập tên (nhiều tên cách nhau bởi dấu phẩy):",
    placeholder="Ví dụ: Alex, Linh, Sakura, Mohammed",
)

if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
    if not names_input.strip():
        st.warning("Vui lòng nhập ít nhất một tên.")
    else:
        client = get_client()
        with st.spinner("Đang dự đoán..."):
            result = predict_via_gemini(client, names_input.strip())
        st.markdown(result)
