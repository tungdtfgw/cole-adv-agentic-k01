"""
Demo 2: Sinh ảnh và chỉnh sửa ảnh multi-turn với Gemini (Nano Banana)
- Tạo ảnh từ prompt
- Chỉnh sửa ảnh qua hội thoại nhiều lượt

Yêu cầu:
    pip install google-genai streamlit Pillow

Chạy:
    export GEMINI_API_KEY="your-api-key"
    streamlit run demo2_image_generation.py
"""

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import base64
import time

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils import get_key

# ============ CONFIG ============
st.set_page_config(
    page_title="Gemini Image Studio",
    page_icon="🎨",
    layout="wide",
)

MODEL = "gemini-3.1-flash-image-preview"

ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "5:4", "4:5", "21:9",
]
RESOLUTIONS = ["512", "1K", "2K", "4K"]


# ============ INIT ============
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())


def init_session_state():
    if "chat" not in st.session_state:
        st.session_state.chat = None
    if "images" not in st.session_state:
        st.session_state.images = []  # List of (prompt, image_bytes, text)
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False


init_session_state()


# ============ IMAGE GENERATION ============
def create_new_chat(aspect_ratio="1:1", resolution="1K"):
    """Create a new chat session for multi-turn image editing."""
    client = get_client()
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
        ),
    )
    return chat


def generate_image(chat, prompt):
    """Send a message to the chat and extract image + text from response."""
    response = chat.send_message(prompt)

    result_text = ""
    result_image = None

    for part in response.parts:
        # Skip thinking parts
        if hasattr(part, "thought") and part.thought:
            continue
        if part.text is not None:
            result_text += part.text
        elif part.inline_data is not None:
            result_image = part.inline_data.data

    return result_image, result_text


def edit_with_uploaded_image(uploaded_image, prompt, aspect_ratio, resolution):
    """One-shot edit: upload an image and apply edits."""
    client = get_client()
    img = Image.open(uploaded_image)

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
        ),
    )

    result_text = ""
    result_image = None

    for part in response.parts:
        if hasattr(part, "thought") and part.thought:
            continue
        if part.text is not None:
            result_text += part.text
        elif part.inline_data is not None:
            result_image = part.inline_data.data

    return result_image, result_text


# ============ UI ============
st.title("🎨 Gemini Image Studio")
st.caption("Sinh ảnh & chỉnh sửa ảnh multi-turn với Nano Banana (Gemini 3.1 Flash Image)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt ảnh")

    mode = st.radio(
        "Chế độ:",
        ["✨ Tạo ảnh mới (multi-turn)", "🖌️ Chỉnh sửa ảnh có sẵn"],
    )

    aspect_ratio = st.selectbox("Tỷ lệ khung hình:", ASPECT_RATIOS, index=0)
    resolution = st.selectbox("Độ phân giải:", RESOLUTIONS, index=1)

    st.divider()

    if mode == "✨ Tạo ảnh mới (multi-turn)":
        if st.button("🔄 Bắt đầu phiên mới", use_container_width=True):
            st.session_state.chat = None
            st.session_state.images = []
            st.session_state.chat_started = False
            st.rerun()

    st.markdown(
        """
    **Hướng dẫn:**
    
    *Chế độ Tạo ảnh mới:*
    - Nhập prompt → sinh ảnh
    - Tiếp tục nhập prompt để chỉnh sửa
    - Mỗi lượt dựa trên ảnh trước đó
    
    *Chế độ Chỉnh sửa:*
    - Upload ảnh gốc
    - Nhập yêu cầu chỉnh sửa
    """
    )

# ============ MODE 1: Multi-turn generation ============
if mode == "✨ Tạo ảnh mới (multi-turn)":

    # Display conversation history
    for i, (prompt, img_bytes, text) in enumerate(st.session_state.images):
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            if text:
                st.caption(text)
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
                st.download_button(
                    f"💾 Tải ảnh #{i+1}",
                    data=img_bytes,
                    file_name=f"gemini_image_{i+1}.png",
                    mime="image/png",
                    key=f"dl_{i}",
                )

    # Chat input
    prompt = st.chat_input("Nhập prompt tạo/chỉnh sửa ảnh...")

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Create chat if needed
        if st.session_state.chat is None:
            st.session_state.chat = create_new_chat(aspect_ratio, resolution)
            st.session_state.chat_started = True

        # Generate
        with st.chat_message("assistant"):
            with st.spinner("Gemini đang xử lý..."):
                try:
                    img_bytes, text = generate_image(st.session_state.chat, prompt)

                    if text:
                        st.caption(text)
                    if img_bytes:
                        st.image(img_bytes, use_container_width=True)

                    st.session_state.images.append((prompt, img_bytes, text))

                    if img_bytes:
                        st.download_button(
                            f"💾 Tải ảnh #{len(st.session_state.images)}",
                            data=img_bytes,
                            file_name=f"gemini_image_{len(st.session_state.images)}.png",
                            mime="image/png",
                            key=f"dl_new_{len(st.session_state.images)}",
                        )
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # Empty state
    if not st.session_state.images:
        st.markdown(
            """
        <div style="text-align: center; padding: 80px 20px; color: #888;">
            <h3>✨ Nhập prompt để bắt đầu tạo ảnh</h3>
            <p>Ví dụ: "A cute cartoon cat wearing a wizard hat, digital art style"</p>
            <p>Sau đó tiếp tục nhập prompt để chỉnh sửa ảnh</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ============ MODE 2: Edit uploaded image ============
else:
    col_upload, col_result = st.columns(2)

    with col_upload:
        st.subheader("📤 Ảnh gốc")
        uploaded_file = st.file_uploader(
            "Upload ảnh cần chỉnh sửa",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)

    with col_result:
        st.subheader("🖼️ Kết quả")

        edit_prompt = st.text_area(
            "Yêu cầu chỉnh sửa:",
            placeholder="Ví dụ: Add a beautiful sunset sky in the background",
            height=100,
        )

        if st.button("🚀 Chỉnh sửa", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("Vui lòng upload ảnh trước!")
            elif not edit_prompt.strip():
                st.warning("Vui lòng nhập yêu cầu chỉnh sửa!")
            else:
                with st.spinner("Gemini đang chỉnh sửa ảnh..."):
                    try:
                        img_bytes, text = edit_with_uploaded_image(
                            uploaded_file, edit_prompt, aspect_ratio, resolution
                        )
                        if text:
                            st.caption(text)
                        if img_bytes:
                            st.image(img_bytes, use_container_width=True)
                            st.download_button(
                                "💾 Tải ảnh đã chỉnh sửa",
                                data=img_bytes,
                                file_name="edited_image.png",
                                mime="image/png",
                            )
                        else:
                            st.warning("Không nhận được ảnh từ Gemini. Hãy thử lại với prompt khác.")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
