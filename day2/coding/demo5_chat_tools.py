"""
=============================================================================
BÀI MẪU 5: Chat với Function Calling — Cat Fact + Genderize
=============================================================================
Công nghệ: Python + Streamlit + Gemini Function Calling
Mô tả:
  - Chat tự do với Gemini
  - Gemini tự quyết định gọi tool nào (hoặc không gọi) dựa trên ngữ cảnh
  - 2 tools: get_cat_fact (Cat Fact API), predict_gender (Genderize API)

Cài đặt:
  pip install google-genai streamlit requests

Chạy:
  export GEMINI_API_KEY="your-api-key"
  streamlit run demo5_chat_tools.py
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
st.set_page_config(page_title="Chat + Tools", page_icon="💬")

MODEL = "gemini-2.5-flash"

# ===== FUNCTION DECLARATIONS =====
DECLARATIONS = [
    {
        "name": "get_cat_fact",
        "description": "Get a random fun fact about cats from Cat Fact API. "
                       "Use when the user asks about cats or wants a cat fact.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "predict_gender",
        "description": "Predict gender (male/female) based on a person's name "
                       "using Genderize.io API. Use when the user asks about "
                       "a name's gender or gives names to predict.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's name to predict gender for.",
                }
            },
            "required": ["name"],
        },
    },
]

# ===== FUNCTION IMPLEMENTATIONS =====
FUNCTIONS = {
    "get_cat_fact": lambda **_: (
        requests.get("https://catfact.ninja/fact", timeout=10).json()
    ),
    "predict_gender": lambda name: (
        requests.get("https://api.genderize.io", params={"name": name}, timeout=10).json()
    ),
}


# ===== GEMINI =====
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())


def get_config():
    return types.GenerateContentConfig(
        system_instruction="Bạn là trợ lý thân thiện, trả lời bằng tiếng Việt. "
                           "Khi cần thông tin về mèo hoặc dự đoán giới tính từ tên, "
                           "hãy sử dụng các tools được cung cấp.",
        tools=[types.Tool(function_declarations=DECLARATIONS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def chat_with_tools(client, messages):
    """Gửi tin nhắn, xử lý function calling nếu có, trả về text cuối."""
    config = get_config()

    response = client.models.generate_content(
        model=MODEL, contents=messages, config=config,
    )

    # Nếu model không gọi tool → trả text luôn
    function_calls = response.function_calls
    if not function_calls:
        return response.text, response.candidates[0].content

    # Chạy tất cả tools được gọi
    messages.append(response.candidates[0].content)

    function_response_parts = []
    for fc in function_calls:
        fn = FUNCTIONS.get(fc.name)
        result = fn(**fc.args) if fn else {"error": f"Unknown function: {fc.name}"}
        function_response_parts.append(
            types.Part.from_function_response(name=fc.name, response={"result": result})
        )

    messages.append(types.Content(role="user", parts=function_response_parts))

    # Gửi kết quả lại → nhận câu trả lời cuối
    final_response = client.models.generate_content(
        model=MODEL, contents=messages, config=config,
    )
    return final_response.text, final_response.candidates[0].content


# ===== SESSION STATE =====
if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {"role", "content"} for display
    st.session_state.api_messages = []  # list of types.Content for API

# ===== UI =====
st.title("💬 Chat + Tools")
st.caption("Chat tự do — Gemini tự gọi Cat Fact / Genderize API khi cần")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Nhắn gì đó..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to API messages
    st.session_state.api_messages.append(
        types.Content(role="user", parts=[types.Part(text=prompt)])
    )

    # Call Gemini
    client = get_client()
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            text, assistant_content = chat_with_tools(
                client, list(st.session_state.api_messages)
            )

        st.markdown(text)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": text})
    st.session_state.api_messages.append(assistant_content)

# Sidebar
with st.sidebar:
    st.markdown("""
    **Tools có sẵn:**
    - 🐱 **Cat Fact** — hỏi về mèo
    - 🧑 **Genderize** — dự đoán giới tính từ tên

    **Thử hỏi:**
    - "Kể tôi nghe về mèo đi"
    - "Tên Sakura là nam hay nữ?"
    - "So sánh Alex và Linh xem tên nào thiên về nữ hơn"
    """)
    if st.button("🔄 Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.api_messages = []
        st.rerun()
