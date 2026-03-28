"""
MCP Client - Streamlit Chat App with Gemini API (google-genai SDK)
Connects to MCP server to query Supabase data via chat interface.
Supports data tables and charts.
"""

import os
import sys
import json
import asyncio
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google import genai
from google.genai import types

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import get_key

MODEL = "gemini-2.5-flash"
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

SYSTEM_INSTRUCTION = (
    "You are a helpful data analyst assistant connected to a Supabase Postgres database "
    "with 3 tables: orders, returns, and people (Superstore dataset).\n\n"
    "IMPORTANT RULES:\n"
    "1. Always call get_schema_info first if you haven't seen the schema yet, to learn the table "
    "structures, column names/types, and sample values.\n"
    "2. Use execute_select to run any SELECT SQL query. You can use JOINs, GROUP BY, aggregations, "
    "subqueries, window functions — any valid PostgreSQL SELECT syntax.\n"
    "3. When the user asks for charts/visualizations, write an appropriate GROUP BY query, "
    "then format your response with a JSON block for the client to render:\n\n"
    "   ```chart\n"
    '   {"chart_type": "bar|line|pie|scatter", "data": [...], "x": "column_name", "y": "column_name", "title": "Chart Title"}\n'
    "   ```\n\n"
    "4. When returning tabular data, include a JSON block:\n"
    "   ```table\n"
    '   [{"col1": "val1", ...}, ...]\n'
    "   ```\n\n"
    "5. Keep text responses concise. Let the data speak.\n"
    "6. Always respond in the same language the user uses.\n"
    "7. For date columns, the format is YYYY-MM-DD.\n"
    "8. Only SELECT queries are allowed. No INSERT/UPDATE/DELETE.\n"
)


# ── MCP Client Class ────────────────────────────────────────────────

class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    async def connect(self):
        """Connect to the MCP server."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT],
            env={**os.environ},
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = response.tools

    def get_gemini_declarations(self) -> list[dict]:
        """Convert MCP tools to Gemini function declarations."""
        declarations = []
        for tool in self.tools:
            schema = tool.inputSchema
            properties = {}
            for prop_name, prop_def in schema.get("properties", {}).items():
                prop_type = prop_def.get("type", "string")
                type_map = {
                    "string": "STRING",
                    "integer": "INTEGER",
                    "number": "NUMBER",
                    "boolean": "BOOLEAN",
                }
                properties[prop_name] = {
                    "type": type_map.get(prop_type, "STRING"),
                    "description": prop_def.get("description", ""),
                }
            declarations.append({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": schema.get("required", []),
                },
            })
        return declarations

    async def call_tool(self, name: str, args: dict) -> str:
        """Call an MCP tool and return the result as string."""
        result = await self.session.call_tool(name, args)
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
        return "\n".join(texts)

    async def cleanup(self):
        await self.exit_stack.aclose()


# ── Async Helper ─────────────────────────────────────────────────────

def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# ── Initialize MCP Client ───────────────────────────────────────────

def init_mcp_client():
    if "mcp_client" not in st.session_state:
        loop = get_or_create_event_loop()
        client = MCPClient()
        loop.run_until_complete(client.connect())
        st.session_state.mcp_client = client
        st.session_state.event_loop = loop
    return st.session_state.mcp_client, st.session_state.event_loop


# ── Chat with Gemini + MCP Tools ────────────────────────────────────

def chat_with_tools(user_message: str, mcp_client: MCPClient, loop: asyncio.AbstractEventLoop) -> str:
    """Send user message to Gemini, handle tool calls via MCP, return final response."""

    client = genai.Client(api_key=get_key())
    declarations = mcp_client.get_gemini_declarations()

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Build API messages from session state
    if "api_messages" not in st.session_state:
        st.session_state.api_messages = []

    st.session_state.api_messages.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    messages = list(st.session_state.api_messages)

    max_iterations = 10
    final_text = ""
    assistant_content = None

    for _ in range(max_iterations):
        response = client.models.generate_content(
            model=MODEL, contents=messages, config=config,
        )

        function_calls = response.function_calls
        if not function_calls:
            final_text = response.text or ""
            assistant_content = response.candidates[0].content
            break

        # Append model response with function calls
        messages.append(response.candidates[0].content)

        # Execute tools via MCP
        function_response_parts = []
        for fc in function_calls:
            tool_args = dict(fc.args) if fc.args else {}
            # Clean up empty/zero args
            clean_args = {k: v for k, v in tool_args.items() if v is not None and v != ""}

            with st.spinner(f"🔧 {fc.name}..."):
                result_text = loop.run_until_complete(
                    mcp_client.call_tool(fc.name, clean_args)
                )

            function_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result_text}
                )
            )

        messages.append(types.Content(role="user", parts=function_response_parts))

    # Save conversation state
    if assistant_content:
        st.session_state.api_messages.append(assistant_content)

    return final_text


# ── Render Response ──────────────────────────────────────────────────

def render_response(text: str):
    """Parse response text and render charts/tables if present."""

    chart_pattern = r"```chart\s*\n(.*?)\n```"
    chart_matches = re.findall(chart_pattern, text, re.DOTALL)

    table_pattern = r"```table\s*\n(.*?)\n```"
    table_matches = re.findall(table_pattern, text, re.DOTALL)

    clean_text = re.sub(r"```(?:chart|table)\s*\n.*?\n```", "", text, flags=re.DOTALL).strip()

    if clean_text:
        st.markdown(clean_text)

    for chart_json in chart_matches:
        try:
            chart_data = json.loads(chart_json)
            df = pd.DataFrame(chart_data["data"])
            chart_type = chart_data.get("chart_type", "bar")
            title = chart_data.get("title", "")
            x = chart_data.get("x", df.columns[0])
            y = chart_data.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])

            chart_funcs = {
                "bar": px.bar, "line": px.line,
                "pie": lambda df, x, y, title: px.pie(df, names=x, values=y, title=title),
                "scatter": px.scatter,
            }
            func = chart_funcs.get(chart_type, px.bar)
            fig = func(df, x=x, y=y, title=title)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    for table_json in table_matches:
        try:
            df = pd.DataFrame(json.loads(table_json))
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Table error: {e}")


# ── Streamlit UI ─────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="MCP Supabase Chat", page_icon="🗄️", layout="wide")
    st.title("MCP Supabase Explorer")
    st.caption("Chat with your Supabase data using Gemini AI + MCP Protocol")

    with st.sidebar:
        st.header("About")
        st.markdown("""
        **MCP Server** kết nối Supabase (Superstore dataset):
        - **orders** — 9,994 đơn hàng
        - **returns** — 296 đơn trả lại
        - **people** — 4 quản lý vùng

        **Thử hỏi:**
        - "Tổng doanh thu theo danh mục"
        - "Top 10 khách hàng có lợi nhuận cao nhất"
        - "Vẽ biểu đồ tròn đơn hàng theo vùng"
        - "Sản phẩm nào bị trả lại nhiều nhất?"
        """)
        if st.button("🔄 Xóa hội thoại", use_container_width=True):
            st.session_state.messages = []
            st.session_state.api_messages = []
            if "gemini_chat" in st.session_state:
                del st.session_state.gemini_chat
            st.rerun()

    try:
        mcp_client, loop = init_mcp_client()
        st.sidebar.success(f"MCP Connected ({len(mcp_client.tools)} tools)")
    except Exception as e:
        import traceback
        st.error(f"MCP connection failed: {e}")
        st.code(traceback.format_exc(), language="text")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_response(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Hỏi về dữ liệu Supabase..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response_text = chat_with_tools(prompt, mcp_client, loop)
                render_response(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
