"""
Demo 3: Sinh đề nghe IELTS tự động với Gemini API
- Bước 1: Gemini sinh transcript đề nghe IELTS (hội thoại 2 người)
- Bước 2: Gemini TTS đọc transcript thành audio WAV (multi-speaker)
- Bước 3: Gemini sinh câu hỏi trắc nghiệm từ transcript

Yêu cầu:
    pip install google-genai streamlit

Chạy:
    export GEMINI_API_KEY="your-api-key"
    streamlit run tut03_ielts_listening.py
"""

import streamlit as st
from google import genai
from google.genai import types
import json
import wave
import io
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key

# ============ CONFIG ============
st.set_page_config(
    page_title="IELTS Listening Generator",
    page_icon="🎧",
    layout="wide",
)

TEXT_MODEL = "gemini-3-flash-preview"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

SPEAKER_1_VOICE = "Puck"
SPEAKER_2_VOICE = "Kore"

TOPICS = [
    "booking a guided city tour at a travel agency",
    "registering for a gym membership",
    "making a reservation at a restaurant for a group dinner",
    "enquiring about a language course at a school",
    "reporting a lost item at a hotel reception",
]


# ============ GEMINI CLIENT ============
@st.cache_resource
def get_client():
    return genai.Client(api_key=get_key())


# ============ STEP 1: Generate Transcript ============
def generate_transcript(client, topic):
    """Gemini sinh transcript hội thoại IELTS Listening Section 1."""
    prompt = f"""You are an IELTS listening test creator. Generate a realistic
IELTS Listening Section 1 transcript.

Topic: {topic}

Requirements:
1. Two speakers: "Agent" (a staff/receptionist) and "Caller" (a customer/visitor).
2. The conversation should be 150-200 words, natural and clear.
3. Include specific details that can be tested: names, numbers, dates,
   addresses, prices, times.
4. The conversation should flow naturally with greetings and closings.
5. Format each line as "Speaker: dialogue" (one speaker per line).

Output ONLY the transcript, no explanations or stage directions.
Keep the language at IELTS Band 5-6 level (clear, not too complex)."""

    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    return response.text.strip()


# ============ STEP 2: TTS Multi-speaker ============
def generate_audio(client, transcript):
    """Gemini TTS chuyển transcript thành audio WAV multi-speaker."""
    tts_prompt = f"""Read the following conversation naturally and clearly.
Speak at a moderate pace suitable for an English listening test.
Pause briefly between speaker turns.

{transcript}"""

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=tts_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Agent",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=SPEAKER_1_VOICE,
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Caller",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=SPEAKER_2_VOICE,
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
    )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data

    # Convert PCM to WAV in memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ============ STEP 3: Generate Questions ============
def generate_questions(client, transcript, num_questions=5):
    """Gemini sinh câu hỏi trắc nghiệm IELTS từ transcript."""
    prompt = f"""Based on this IELTS Listening transcript, create {num_questions}
multiple-choice questions (A, B, C) that test comprehension of specific details.

Transcript:
{transcript}

Requirements:
1. Questions should test factual details (names, numbers, dates, places, prices).
2. Each question has exactly 3 options (A, B, C) with only ONE correct answer.
3. Distractors should be plausible but clearly wrong based on the transcript.
4. Questions should follow the order of information in the transcript.

Return ONLY a valid JSON array with this structure:
[
  {{
    "question": "What is the caller's name?",
    "options": {{"A": "John Smith", "B": "James Brown", "C": "Jack White"}},
    "answer": "A",
    "explanation": "The caller introduces himself as John Smith."
  }}
]"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text.strip())


# ============ UI ============
st.title("🎧 IELTS Listening Test Generator")
st.caption("Sinh đề nghe IELTS tự động với Gemini API (Transcript → Audio → Questions)")

# --- Session state ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
    st.session_state.audio = None
    st.session_state.questions = None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    topic_choice = st.selectbox("Chủ đề:", TOPICS)
    custom_topic = st.text_input("Hoặc nhập chủ đề tùy chọn:")
    num_questions = st.slider("Số câu hỏi:", 3, 10, 5)

    if st.button("🔄 Tạo đề mới", use_container_width=True):
        st.session_state.transcript = None
        st.session_state.audio = None
        st.session_state.questions = None
        st.rerun()

# --- Main flow ---
topic = custom_topic.strip() if custom_topic.strip() else topic_choice
client = get_client()

# Step 1: Transcript
if st.session_state.transcript is None:
    if st.button("▶️ Bắt đầu sinh đề", type="primary", use_container_width=True):
        with st.status("Đang sinh đề IELTS...", expanded=True) as status:
            # Step 1
            st.write("📝 Bước 1: Sinh transcript...")
            st.session_state.transcript = generate_transcript(client, topic)

            # Step 2
            st.write("🎙️ Bước 2: Chuyển thành audio (TTS multi-speaker)...")
            st.session_state.audio = generate_audio(client, st.session_state.transcript)

            # Step 3
            st.write(f"❓ Bước 3: Sinh {num_questions} câu hỏi...")
            st.session_state.questions = generate_questions(
                client, st.session_state.transcript, num_questions
            )

            status.update(label="Hoàn thành!", state="complete")
        st.rerun()
else:
    st.info(f"**Chủ đề:** {topic}")

# --- Display results ---
if st.session_state.transcript:
    tab_exam, tab_transcript, tab_answers = st.tabs(
        ["📝 Làm bài", "📄 Transcript", "✅ Đáp án"]
    )

    # Tab 1: Exam
    with tab_exam:
        st.subheader("🎵 Audio")
        st.audio(st.session_state.audio, format="audio/wav")

        st.divider()
        st.subheader("📝 Questions")

        user_answers = {}
        for i, q in enumerate(st.session_state.questions):
            options = [f"{k}. {v}" for k, v in q["options"].items()]
            choice = st.radio(
                f"**Q{i+1}.** {q['question']}",
                options,
                index=None,
                key=f"q_{i}",
            )
            if choice:
                user_answers[i] = choice[0]  # "A", "B", or "C"

        if st.button("📊 Chấm điểm", type="primary"):
            correct = sum(
                1 for i, q in enumerate(st.session_state.questions)
                if user_answers.get(i) == q["answer"]
            )
            total = len(st.session_state.questions)
            st.success(f"Kết quả: **{correct}/{total}** câu đúng")

            for i, q in enumerate(st.session_state.questions):
                ans = user_answers.get(i)
                if ans == q["answer"]:
                    st.write(f"✅ Q{i+1}: {ans} — Đúng")
                elif ans:
                    st.write(f"❌ Q{i+1}: {ans} (đáp án: {q['answer']}) — {q.get('explanation', '')}")
                else:
                    st.write(f"⬜ Q{i+1}: Chưa trả lời (đáp án: {q['answer']})")

    # Tab 2: Transcript
    with tab_transcript:
        st.text(st.session_state.transcript)

    # Tab 3: Answer key
    with tab_answers:
        for i, q in enumerate(st.session_state.questions):
            st.write(f"**{i+1}. {q['answer']}** — {q.get('explanation', '')}")
