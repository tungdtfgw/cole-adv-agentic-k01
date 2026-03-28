"""Streamlit voice chat app with Gemini Live API for crypto assistant.

Embeds a custom JS component that:
- Captures mic audio continuously via Web Audio API
- Streams to Gemini Live API via WebSocket (JS SDK)
- Plays back audio responses in real-time
- Handles function calling for crypto data via FreeCryptoAPI REST
"""

import asyncio
import json

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

from crypto_tools import CryptoToolExecutor, get_crypto_tool_declarations

GEMINI_MODEL = "gemini-3.1-flash-live-preview"

SYSTEM_INSTRUCTION = """Bạn là một trợ lý ảo chuyên về thị trường tiền điện tử (Crypto).
Hãy trả lời ngắn gọn, súc tích và thân thiện.
Bạn có thể cập nhật giá và giải đáp các thắc mắc về blockchain, Bitcoin, Ethereum và các altcoin khác.
Nếu người dùng nói tiếng Việt, hãy trả lời bằng tiếng Việt.
Nếu người dùng nói tiếng Anh, hãy trả lời bằng tiếng Anh.

When users ask about crypto prices, conversions, technical analysis,
Bollinger Bands, or the Fear & Greed Index, use the available tools to get real-time data.
"""


def build_voice_component(gemini_api_key: str, crypto_api_key: str) -> str:
    """Build the HTML/JS component for real-time voice chat."""

    tool_declarations = get_crypto_tool_declarations()
    tools_json = json.dumps(tool_declarations)

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: transparent; }}

  .container {{
    display: flex; flex-direction: column; align-items: center;
    padding: 20px; gap: 20px;
  }}

  .mic-section {{
    display: flex; flex-direction: column; align-items: center; gap: 12px;
  }}

  .mic-btn {{
    width: 80px; height: 80px; border-radius: 50%; border: none;
    cursor: pointer; transition: all 0.3s;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; position: relative;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
  }}
  .mic-btn:hover {{ transform: scale(1.05); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }}
  .mic-btn.active {{
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 0 30px rgba(245, 87, 108, 0.5);
    animation: pulse 1.5s infinite;
  }}
  .mic-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

  @keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 20px rgba(245, 87, 108, 0.4); }}
    50% {{ box-shadow: 0 0 40px rgba(245, 87, 108, 0.7); }}
  }}

  .status {{
    font-size: 14px; color: #888; min-height: 20px; text-align: center;
  }}
  .status.connected {{ color: #10b981; }}
  .status.error {{ color: #ef4444; }}

  .visualizer {{
    display: flex; gap: 3px; align-items: center; height: 40px;
  }}
  .visualizer .bar {{
    width: 4px; background: #667eea; border-radius: 2px;
    transition: height 0.1s; height: 4px;
  }}
  .visualizer.active .bar {{ animation: wave 0.5s ease infinite; }}
  .visualizer.active .bar:nth-child(1) {{ animation-delay: 0s; }}
  .visualizer.active .bar:nth-child(2) {{ animation-delay: 0.1s; }}
  .visualizer.active .bar:nth-child(3) {{ animation-delay: 0.2s; }}
  .visualizer.active .bar:nth-child(4) {{ animation-delay: 0.3s; }}
  .visualizer.active .bar:nth-child(5) {{ animation-delay: 0.4s; }}

  @keyframes wave {{
    0%, 100% {{ height: 8px; }}
    50% {{ height: 32px; }}
  }}

  .transcript-box {{
    width: 100%; max-height: 400px; overflow-y: auto;
    background: #1a1a2e; border-radius: 12px; padding: 16px;
    font-size: 14px; line-height: 1.6;
  }}
  .transcript-box .msg {{
    padding: 8px 12px; margin: 4px 0; border-radius: 8px;
    max-width: 85%;
  }}
  .transcript-box .user {{
    background: #2d2d44; color: #e0e0e0;
    margin-left: auto; text-align: right;
  }}
  .transcript-box .assistant {{
    background: #1e3a5f; color: #e0e0e0;
  }}
  .transcript-box .tool {{
    background: #2d3b2d; color: #a0d0a0;
    font-size: 12px; font-style: italic;
  }}
  .transcript-box .system {{
    color: #888; font-size: 12px; text-align: center;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="mic-section">
    <button class="mic-btn" id="micBtn" onclick="toggleSession()">🎤</button>
    <div class="visualizer" id="visualizer">
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div>
    </div>
    <div class="status" id="status">Click the mic to start</div>
  </div>
  <div class="transcript-box" id="transcript"></div>
</div>

<script type="importmap">
  {{ "imports": {{ "@anthropic-ai/sdk": "https://esm.sh/@anthropic-ai/sdk", "@google/genai": "https://esm.sh/@google/genai" }} }}
</script>
<script type="module">
import {{ GoogleGenAI, Modality }} from "@google/genai";

const GEMINI_API_KEY = "{gemini_api_key}";
const CRYPTO_API_KEY = "{crypto_api_key}";
const CRYPTO_API_BASE = "https://api.freecryptoapi.com/v1";
const TOOLS = {tools_json};

let session = null;
let audioContext = null;
let stream = null;
let processor = null;
let audioQueue = [];
let isPlaying = false;
let playbackCtx = null;
let isActive = false;
let currentUserText = "";
let currentGeminiText = "";

const micBtn = document.getElementById("micBtn");
const statusEl = document.getElementById("status");
const visualizer = document.getElementById("visualizer");
const transcriptEl = document.getElementById("transcript");

// --- Crypto API helpers ---
async function callCryptoAPI(endpoint, params = {{}}) {{
  const url = new URL(CRYPTO_API_BASE + endpoint);
  Object.entries(params).forEach(([k, v]) => {{
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }});
  const resp = await fetch(url, {{
    headers: {{ "Authorization": "Bearer " + CRYPTO_API_KEY, "Accept": "application/json" }}
  }});
  return await resp.json();
}}

async function executeTool(name, args) {{
  try {{
    if (name === "get_crypto_price") return {{ result: JSON.stringify(await callCryptoAPI("/getData", {{ symbol: args.symbols }})) }};
    if (name === "convert_crypto") return {{ result: JSON.stringify(await callCryptoAPI("/getConversion", {{ from: args.from_symbol, to: args.to_symbol, amount: args.amount }})) }};
    if (name === "get_technical_analysis") return {{ result: JSON.stringify(await callCryptoAPI("/getTechnicalAnalysis", {{ symbol: args.symbol }})) }};
    if (name === "get_bollinger_bands") return {{ result: JSON.stringify(await callCryptoAPI("/getBollinger", {{ symbol: args.symbol, days: args.days || 90 }})) }};
    if (name === "get_fear_greed_index") return {{ result: JSON.stringify(await callCryptoAPI("/getFearGreed")) }};
    return {{ error: "Unknown function: " + name }};
  }} catch (e) {{
    return {{ error: e.message }};
  }}
}}

// --- Transcript ---
function addMessage(type, text) {{
  const div = document.createElement("div");
  div.className = "msg " + type;
  if (type === "user") div.textContent = "🧑 " + text;
  else if (type === "assistant") div.textContent = "🤖 " + text;
  else if (type === "tool") div.textContent = "🔧 " + text;
  else {{ div.className = "system"; div.textContent = text; }}
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}}

// --- Audio playback ---
function playNextInQueue() {{
  if (audioQueue.length === 0) {{ isPlaying = false; return; }}
  isPlaying = true;
  const pcmData = audioQueue.shift();

  if (!playbackCtx) playbackCtx = new AudioContext({{ sampleRate: 24000 }});

  const buffer = playbackCtx.createBuffer(1, pcmData.length, 24000);
  const channelData = buffer.getChannelData(0);
  for (let i = 0; i < pcmData.length; i++) channelData[i] = pcmData[i] / 0x7FFF;

  const source = playbackCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackCtx.destination);
  source.onended = () => playNextInQueue();
  source.start();
}}

// --- Session ---
async function startSession() {{
  statusEl.textContent = "Connecting...";
  statusEl.className = "status";

  try {{
    console.log("Connecting to Gemini Live API...");
    const ai = new GoogleGenAI({{ apiKey: GEMINI_API_KEY }});

    session = await ai.live.connect({{
      model: "gemini-3.1-flash-live-preview",
      config: {{
        responseModalities: [Modality.AUDIO],
        speechConfig: {{
          voiceConfig: {{ prebuiltVoiceConfig: {{ voiceName: "Puck" }} }},
        }},
        systemInstruction: `{SYSTEM_INSTRUCTION.replace(chr(96), "").replace(chr(10), " ")}`,
        outputAudioTranscription: {{}},
        inputAudioTranscription: {{}},
        tools: [{{ functionDeclarations: TOOLS }}],
      }},
      callbacks: {{
        onopen: () => {{
          console.log("Gemini Live session opened!");
          statusEl.textContent = "Connected! Start talking...";
          statusEl.className = "status connected";
          isActive = true;
          micBtn.classList.add("active");
          visualizer.classList.add("active");
          addMessage("system", "Session started. Speak naturally!");
          startAudioCapture();
        }},
        onmessage: async (message) => {{
          try {{
          // Tool calls
          if (message.toolCall) {{
            const responses = [];
            for (const fc of message.toolCall.functionCalls) {{
              addMessage("tool", "Calling " + fc.name + "...");
              const result = await executeTool(fc.name, fc.args);
              responses.push({{ name: fc.name, id: fc.id, response: result }});
            }}
            session.sendToolResponse({{ functionResponses: responses }});
            return;
          }}

          const sc = message.serverContent;
          if (!sc) return;

          // Interrupted
          if (sc.interrupted) {{
            audioQueue = [];
            isPlaying = false;
            return;
          }}

          // Input transcription
          if (sc.inputTranscription?.text) {{
            currentUserText += sc.inputTranscription.text;
          }}

          // Output transcription
          if (sc.outputTranscription?.text) {{
            currentGeminiText += sc.outputTranscription.text;
          }}

          // Audio response
          if (sc.modelTurn?.parts) {{
            for (const part of sc.modelTurn.parts) {{
              if (part.inlineData?.data) {{
                const binaryString = atob(part.inlineData.data);
                const bytes = new Int16Array(binaryString.length / 2);
                for (let i = 0; i < bytes.length; i++) {{
                  bytes[i] = (binaryString.charCodeAt(i * 2) & 0xFF) | (binaryString.charCodeAt(i * 2 + 1) << 8);
                }}
                audioQueue.push(bytes);
                if (!isPlaying) playNextInQueue();
              }}
            }}
          }}

          // Turn complete
          if (sc.turnComplete) {{
            if (currentUserText.trim()) addMessage("user", currentUserText.trim());
            if (currentGeminiText.trim()) addMessage("assistant", currentGeminiText.trim());
            currentUserText = "";
            currentGeminiText = "";
          }}
          }} catch (msgErr) {{
            console.error("onmessage error:", msgErr);
            addMessage("system", "Message handling error: " + msgErr.message);
          }}
        }},
        onclose: (e) => {{
          console.log("Session closed:", e);
          addMessage("system", "Connection closed" + (e?.reason ? ": " + e.reason : ""));
          cleanupSession();
        }},
        onerror: (err) => {{
          console.error("Live API Error:", err);
          addMessage("system", "Error: " + JSON.stringify(err));
          statusEl.textContent = "Error — check browser console (F12)";
          statusEl.className = "status error";
        }}
      }}
    }});
  }} catch (error) {{
    console.error("Connection failed:", error);
    statusEl.textContent = "Error: " + error.message;
    statusEl.className = "status error";
  }}
}}

// Called when server closes the connection
function cleanupSession() {{
  isActive = false;
  micBtn.classList.remove("active");
  visualizer.classList.remove("active");
  statusEl.textContent = "Session ended. Click mic to restart.";
  statusEl.className = "status";
  session = null;
  stopAudioCapture();
  audioQueue = [];
  isPlaying = false;
}}

// Called when user clicks stop
function stopSession() {{
  isActive = false;
  micBtn.classList.remove("active");
  visualizer.classList.remove("active");
  statusEl.textContent = "Session ended. Click mic to restart.";
  statusEl.className = "status";
  stopAudioCapture();
  audioQueue = [];
  isPlaying = false;
  if (session) {{
    const s = session;
    session = null;
    try {{ s.close(); }} catch(e) {{ console.log("Close error:", e); }}
  }}
  addMessage("system", "Session ended.");
}}

// --- Base64 encode (safe for large buffers) ---
function arrayBufferToBase64(buffer) {{
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {{
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }}
  return btoa(binary);
}}

// --- Audio capture ---
async function startAudioCapture() {{
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: 16000 }});
    const source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {{
      if (!isActive || !session) return;
      try {{
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {{
          pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
        }}
        const base64Data = arrayBufferToBase64(pcmData.buffer);
        session.sendRealtimeInput({{
          audio: {{ data: base64Data, mimeType: "audio/pcm;rate=16000" }}
        }});
      }} catch (err) {{
        console.error("Audio send error:", err);
      }}
    }};

    source.connect(processor);
    processor.connect(audioContext.destination);
    console.log("Audio capture started");
  }} catch (error) {{
    console.error("Mic error:", error);
    statusEl.textContent = "Mic access denied";
    statusEl.className = "status error";
  }}
}}

function stopAudioCapture() {{
  if (stream) {{ stream.getTracks().forEach(t => t.stop()); stream = null; }}
  if (processor) {{ processor.disconnect(); processor = null; }}
  if (audioContext) {{ audioContext.close(); audioContext = null; }}
}}

// --- Toggle ---
window.toggleSession = () => {{
  if (isActive) stopSession();
  else startSession();
}};
</script>
</body>
</html>
"""


def main():
    st.set_page_config(
        page_title="Crypto Voice Assistant",
        page_icon="🪙",
        layout="wide",
    )

    st.title("🪙 Crypto Voice Assistant")
    st.caption("Real-time voice chat about cryptocurrency — powered by Gemini Live API")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        gemini_key = st.text_input(
            "Gemini API Key",
            value=st.secrets.get("GEMINI_API_KEY", ""),
            type="password",
        )
        crypto_key = st.text_input(
            "FreeCrypto API Key",
            value=st.secrets.get("FREECRYPTO_API_KEY", ""),
            type="password",
        )

        st.divider()
        st.header("Quick Actions (Text Chat)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 BTC Price", use_container_width=True):
                st.session_state["quick_query"] = "What is the current Bitcoin price?"
            if st.button("📈 Top Coins", use_container_width=True):
                st.session_state["quick_query"] = "Show me prices for BTC, ETH, and SOL"
        with col2:
            if st.button("😱 Fear & Greed", use_container_width=True):
                st.session_state["quick_query"] = "What is the current Fear and Greed Index?"
            if st.button("🔍 BTC Analysis", use_container_width=True):
                st.session_state["quick_query"] = "Give me technical analysis for Bitcoin"

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Voice Chat ---
    if gemini_key and gemini_key != "your-gemini-api-key":
        html_content = build_voice_component(gemini_key, crypto_key)
        components.html(html_content, height=600, scrolling=False)
    else:
        st.warning("Enter your Gemini API Key in the sidebar to enable voice chat.")

    # --- Text Chat ---
    st.divider()
    st.subheader("💬 Text Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    quick_query = st.session_state.pop("quick_query", None)
    text_input = st.chat_input("Type your crypto question here...")
    query = quick_query or text_input

    if query and gemini_key and gemini_key != "your-gemini-api-key":
        crypto_executor = CryptoToolExecutor(crypto_key)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("*Thinking...*")
            try:
                client = genai.Client(api_key=gemini_key)
                gemini_text = asyncio.run(
                    _run_text_session(client, query, placeholder, crypto_executor)
                )
                display_text = gemini_text or "*No response*"
                placeholder.markdown(display_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": display_text}
                )
            except Exception as e:
                placeholder.error(f"Error: {e}")


async def _run_text_session(client, text_input, placeholder, crypto_executor):
    """Text-only Gemini Live session."""
    tool_declarations = get_crypto_tool_declarations()
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": types.Content(
            parts=[types.Part(text=SYSTEM_INSTRUCTION)]
        ),
        "tools": [types.Tool(function_declarations=tool_declarations)],
    }

    gemini_text = ""
    async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part(text=text_input)])],
            turn_complete=True,
        )
        async for response in session.receive():
            if response.tool_call:
                function_responses = []
                for fc in response.tool_call.function_calls:
                    placeholder.info(f"🔧 Calling `{fc.name}`...")
                    result = crypto_executor.execute(fc.name, fc.args)
                    function_responses.append(
                        types.FunctionResponse(name=fc.name, id=fc.id, response=result)
                    )
                await session.send_tool_response(function_responses=function_responses)
                continue

            server_content = response.server_content
            if server_content is None:
                continue
            if server_content.model_turn:
                for part in server_content.model_turn.parts:
                    if part.text:
                        gemini_text += part.text
                        placeholder.markdown(gemini_text)
            if server_content.turn_complete:
                break

    return gemini_text


if __name__ == "__main__":
    main()
