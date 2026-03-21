# Hướng dẫn chạy 3 Demo - Xử lý Đa phương tiện với Gemini API

## Yêu cầu chung

```bash
# Cài đặt SDK chính
pip install google-genai

# Thiết lập API key (lấy từ https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-api-key-here"
```

---

## Demo 1: Xử lý tài liệu PDF (Streamlit)

**Chức năng:** Upload PDF → Tóm tắt nội dung / Trích xuất bảng biểu ra Markdown

```bash
pip install streamlit
streamlit run demo1_pdf_processing.py
```

**Models sử dụng:** `gemini-3-flash-preview`

---

## Demo 2: Sinh ảnh & Chỉnh sửa ảnh (Streamlit)

**Chức năng:**
- Tạo ảnh từ prompt (multi-turn: tạo → chỉnh sửa → chỉnh sửa tiếp...)
- Chỉnh sửa ảnh có sẵn (upload ảnh + prompt chỉnh sửa)

```bash
pip install streamlit Pillow
streamlit run demo2_image_generation.py
```

**Models sử dụng:** `gemini-3.1-flash-image-preview` (Nano Banana 2)

---

## Demo 3: Sinh đề nghe IELTS (Python Desktop)

**Chức năng:**
1. Gemini sinh transcript hội thoại IELTS Listening Section 1
2. Gemini TTS đọc transcript thành file audio WAV (2 giọng khác nhau)
3. Gemini sinh câu hỏi trắc nghiệm từ transcript

```bash
python demo3_ielts_listening.py
```

**Output:** Thư mục `ielts_output/` chứa:
- `transcript.txt` — Transcript hội thoại
- `listening_audio.wav` — File audio nghe
- `questions.txt` — Đề thi + đáp án

**Models sử dụng:**
- `gemini-3-flash-preview` (sinh transcript + câu hỏi)
- `gemini-2.5-flash-preview-tts` (text-to-speech)
