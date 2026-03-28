"""
MCP Server - Supabase Connector
Exposes 3 tools: test_connection, get_schema_info, and execute_select.
Uses Supabase REST API + RPC (no direct Postgres connection needed).
"""

import os
import sys
import json
import logging
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pathlib import Path

# Logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(
    level=logging.DEBUG,
    format="[MCP-SERVER] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
logger.info(f"Loading .env from: {ENV_PATH} (exists: {ENV_PATH.exists()})")
load_dotenv(ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

logger.info(f"SUPABASE_URL loaded: {'yes' if SUPABASE_URL else 'NO'}")
logger.info(f"SUPABASE_SERVICE_ROLE_KEY loaded: {'yes' if SUPABASE_KEY else 'NO'}")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

mcp = FastMCP("supabase-demo")


async def _rpc(function_name: str, params: dict) -> dict | list:
    """Call a Supabase RPC function."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{function_name}",
            headers=HEADERS,
            json=params,
        )
        resp.raise_for_status()
        return resp.json()


# ── Tool: Test Connection ────────────────────────────────────────────

@mcp.tool()
async def test_connection() -> str:
    """Test the database connection and return server info.
    Use this to verify the MCP server can reach Supabase Postgres.
    """
    try:
        result = await _rpc("execute_sql", {"query": "SELECT version() as version, current_database() as db"})
        if result and len(result) > 0:
            return json.dumps({"status": "connected", **result[0]})
        return json.dumps({"status": "connected", "detail": "OK"})
    except Exception as e:
        logger.error(f"test_connection failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool: Get Schema Info ───────────────────────────────────────────

@mcp.tool()
async def get_schema_info() -> str:
    """Get schema information for all tables in the public schema.
    Returns table names, columns, data types, row counts, and sample values.
    Always call this first to understand the database before writing queries.
    """
    try:
        # Get tables and columns
        columns = await _rpc("execute_sql", {"query": """
            SELECT
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable
            FROM information_schema.tables t
            JOIN information_schema.columns c
                ON t.table_name = c.table_name AND t.table_schema = c.table_schema
            WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name, c.ordinal_position
        """})

        # Get row counts
        tables = list(set(row["table_name"] for row in columns))
        row_counts = {}
        for table in tables:
            count_result = await _rpc("execute_sql", {
                "query": f"SELECT COUNT(*) as cnt FROM public.\"{table}\""
            })
            row_counts[table] = count_result[0]["cnt"] if count_result else 0

        # Get sample values for text columns
        sample_values = {}
        for table in tables:
            sample_values[table] = {}
            text_cols = [
                row["column_name"] for row in columns
                if row["table_name"] == table and row["data_type"] in ("text", "character varying")
            ]
            for col in text_cols[:10]:
                vals = await _rpc("execute_sql", {
                    "query": f"SELECT DISTINCT \"{col}\" as val FROM public.\"{table}\" WHERE \"{col}\" IS NOT NULL LIMIT 5"
                })
                sample_values[table][col] = [v["val"] for v in vals]

        # Build schema dict
        schema = {}
        for row in columns:
            table = row["table_name"]
            if table not in schema:
                schema[table] = {
                    "row_count": row_counts.get(table, 0),
                    "columns": [],
                    "sample_values": sample_values.get(table, {}),
                }
            schema[table]["columns"].append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
            })

        return json.dumps(schema, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"get_schema_info failed: {e}")
        return json.dumps({"error": str(e)})


# ── Tool: Execute SELECT ────────────────────────────────────���────────

@mcp.tool()
async def execute_select(sql: str) -> str:
    """Execute a read-only SQL SELECT query against the Supabase Postgres database.

    Args:
        sql: A SELECT SQL query. Only SELECT statements are allowed.
             Examples:
             - SELECT category, SUM(sales) as total_sales FROM orders GROUP BY category ORDER BY total_sales DESC
             - SELECT o.order_id, o.customer_name, o.sales FROM orders o JOIN returns r ON o.order_id = r.updated_order_returns
             - SELECT region, COUNT(*) as order_count FROM orders GROUP BY region
             - SELECT * FROM people
             - SELECT * FROM orders LIMIT 10

    Returns:
        JSON with rows array and count. Limited to 500 rows max.
    """
    # Safety: only allow SELECT
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed."})

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    for keyword in forbidden:
        if f" {keyword} " in f" {stripped} " or stripped.startswith(keyword):
            return json.dumps({"error": f"Forbidden keyword: {keyword}. Only SELECT queries are allowed."})

    # Add LIMIT if not present
    if "LIMIT" not in stripped:
        sql = sql.rstrip(";") + " LIMIT 500"

    try:
        result = await _rpc("execute_sql", {"query": sql})
        if isinstance(result, dict) and "error" in result:
            return json.dumps(result)

        rows = result if isinstance(result, list) else []
        return json.dumps({
            "rows": rows[:500],
            "count": len(rows),
            "truncated": len(rows) > 500,
        }, ensure_ascii=False, default=str)
    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        logger.error(f"execute_select HTTP error: {error_body}")
        return json.dumps({"error": error_body})
    except Exception as e:
        logger.error(f"execute_select failed: {e}")
        return json.dumps({"error": str(e)})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
