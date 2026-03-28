"""
Ví dụ 1: Implicit Caching - Hỏi đáp PDF
- Upload file PDF lên Gemini
- Người dùng hỏi đáp về nội dung PDF (stateless, gửi full PDF mỗi lần)
- Gemini tự động implicit cache nội dung trùng lặp từ lần gửi thứ 2
- Kết thúc in report token usage

Cài đặt:
  pip install google-genai python-dotenv

Chạy:
  python tut01_basic_caching.py
"""
import os
import sys
import pathlib
from google import genai
from google.genai import types

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key

# ==================================================
# Khởi tạo
# ==================================================
client = genai.Client(api_key=get_key())
MODEL = "gemini-3-flash-preview"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thông minh. "
    "Trả lời câu hỏi dựa HOÀN TOÀN vào tài liệu PDF được cung cấp. "
    "Trả lời bằng tiếng Việt. "
    "Nếu không tìm thấy thông tin trong tài liệu, hãy nói rõ."
)

# ==================================================
# Upload file PDF
# ==================================================
pdf_path = os.path.join(os.path.dirname(__file__), "example.pdf")
if not os.path.exists(pdf_path):
    print(f"Không tìm thấy file: {pdf_path}")
    print("Vui lòng đặt file example.pdf vào cùng thư mục với script này.")
    sys.exit(1)

print(f"Đang upload: {pdf_path}")
gemini_file = client.files.upload(
    file=pathlib.Path(pdf_path),
    config=dict(mime_type="application/pdf"),
)
print(f"Upload thành công: {gemini_file.name}")

# ==================================================
# Hỏi đáp interactive
# ==================================================
print(f"\n{'='*60}")
print("HỎI ĐÁP VỀ NỘI DUNG PDF (Implicit Caching)")
print(f"{'='*60}")
print("Gõ câu hỏi và nhấn Enter. Gõ 'q' để kết thúc.\n")

qa_log = []

while True:
    question = input("Câu hỏi: ").strip()
    if not question:
        continue
    if question.lower() == 'q':
        break

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=gemini_file.uri,
                        mime_type="application/pdf",
                    ),
                    types.Part(text=f"\n{SYSTEM_INSTRUCTION}\n\nCâu hỏi: {question}"),
                ]
            )
        ]
    )

    print(f"\nTrả lời: {response.text}\n")

    usage = response.usage_metadata
    qa_log.append({
        "question": question,
        "input_tokens": usage.prompt_token_count,
        "cached_tokens": getattr(usage, 'cached_content_token_count', 0) or 0,
        "output_tokens": usage.candidates_token_count,
    })

# ==================================================
# Báo cáo
# ==================================================
if qa_log:
    print(f"\n{'='*60}")
    print("BÁO CÁO TOKEN USAGE")
    print(f"{'='*60}\n")

    total_input = 0
    total_cached = 0
    total_output = 0

    for i, entry in enumerate(qa_log, 1):
        print(f"  Q{i}: {entry['question'][:50]}")
        print(f"      Input tokens:   {entry['input_tokens']:>8,}")
        print(f"      Cached tokens:  {entry['cached_tokens']:>8,}")
        print(f"      Output tokens:  {entry['output_tokens']:>8,}")
        print()

        total_input += entry['input_tokens']
        total_cached += entry['cached_tokens']
        total_output += entry['output_tokens']

    print(f"  {'─'*50}")
    print(f"  TỔNG CỘNG ({len(qa_log)} câu hỏi):")
    print(f"      Input tokens:   {total_input:>8,}")
    print(f"      Cached tokens:  {total_cached:>8,}")
    print(f"      Output tokens:  {total_output:>8,}")
else:
    print("\n(Không có câu hỏi nào được hỏi)")

print("\nHoàn thành!")
