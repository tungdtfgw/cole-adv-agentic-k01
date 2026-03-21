"""
=============================================================================
BÀI MẪU 3: Function Calling — Gọi Cat Fact API
=============================================================================
Công nghệ: Python + Streamlit + Gemini Function Calling
Mô tả:
  - Khai báo hàm get_cat_fact() gọi API https://catfact.ninja/fact
  - Model tự quyết định khi nào gọi hàm (Function Calling)
  - Hỗ trợ hiển thị tiếng Anh hoặc dịch sang tiếng Việt qua Gemini

Cài đặt:
  pip install google-genai streamlit requests

Chạy:
  export GEMINI_API_KEY="your-api-key"
  streamlit run demo3_catfact.py
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
st.set_page_config(page_title="Cat Fact", page_icon="🐱")

MODEL = "gemini-2.5-flash"

# Function Declaration — mô tả hàm cho model biết
get_cat_fact_declaration = {
    "name": "get_cat_fact",
    "description": "Get a random fun fact about cats from Cat Fact API.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# Hàm thực tế gọi API
def get_cat_fact() -> dict:
    """Gọi Cat Fact API và trả về một fun fact ngẫu nhiên về mèo."""
    response = requests.get("https://catfact.ninja/fact", timeout=10)
    response.raise_for_status()
    data = response.json()
    return {"fact": data["fact"]}


# Gemini client
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())


def get_cat_fact_via_gemini(client, language):
    """Dùng Function Calling để lấy cat fact, dịch nếu cần."""
    tools = types.Tool(function_declarations=[get_cat_fact_declaration])

    if language == "Tiếng Việt":
        user_prompt = "Kể cho tôi một sự thật thú vị về mèo, trả lời bằng tiếng Việt."
    else:
        user_prompt = "Tell me a fun fact about cats."

    config = types.GenerateContentConfig(tools=[tools])

    # Bước 1: Gửi prompt → Model trả về function_call
    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    response = client.models.generate_content(
        model=MODEL, contents=contents, config=config,
    )

    function_call = response.candidates[0].content.parts[0].function_call
    if not function_call:
        return response.text

    # Bước 2: Chạy hàm thực tế
    result = get_cat_fact()

    # Bước 3: Gửi kết quả lại cho model → Nhận câu trả lời cuối
    function_response_part = types.Part.from_function_response(
        name=function_call.name,
        response={"result": result},
    )
    contents.append(response.candidates[0].content)
    contents.append(types.Content(role="user", parts=[function_response_part]))

    final_response = client.models.generate_content(
        model=MODEL, contents=contents, config=config,
    )
    return final_response.text


# ===== UI =====
st.title("🐱 Cat Fact")
st.caption("Function Calling demo — Gemini gọi Cat Fact API và trả lời")

language = st.radio("Ngôn ngữ:", ["English", "Tiếng Việt"], horizontal=True)

if st.button("🐾 Tìm hiểu về mèo", type="primary", use_container_width=True):
    client = get_client()
    with st.spinner("Đang hỏi mèo..."):
        result = get_cat_fact_via_gemini(client, language)
    st.info(result)
