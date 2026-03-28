# MCP Demo - Supabase Chat with Gemini

Demo ứng dụng MCP Client + MCP Server kết nối Supabase qua giao diện Streamlit chat.

## Architecture

```
User <-> Streamlit (MCP Client + Gemini) <-> MCP Server <-> Supabase REST API
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add to `.env` (project root):
```
GEMINI_API_KEY=your_key
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

3. Run:
```bash
streamlit run day3/mcp/client.py
```

## MCP Server Tools

| Tool | Description |
|------|-------------|
| `get_schema_info` | Get schema of all 3 tables |
| `query_orders` | Query orders with filters |
| `query_returns` | Query returns with filters |
| `query_people` | Query people (regional managers) |
| `aggregate_orders` | Group-by aggregation for charts |

## Dataset

Superstore dataset on Supabase:
- **orders** (9,994 rows) - Sales transactions
- **returns** (296 rows) - Returned orders
- **people** (4 rows) - Regional managers
