import sys
import os
import ast
import json
import re
import tempfile
import requests
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import faiss
import networkx as nx
from pyvis.network import Network
from fastembed import TextEmbedding

from google import genai
from google.genai import types

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import get_key

MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 10

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_data")
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")

# ============ GitHub Fetch ============

def fetch_github_tree(owner, repo, branch="main"):
    """Fetch the full file tree of a GitHub repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        # Try 'master' branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("tree", [])


def fetch_file_content(owner, repo, path, branch="main"):
    """Fetch raw content of a single file from GitHub."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{path}"
        resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        return resp.text
    return None


CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".rs", ".cpp", ".c", ".h", ".cs"}


def fetch_repo_files(owner, repo):
    """Fetch all code files from a GitHub repo."""
    tree = fetch_github_tree(owner, repo)
    files = {}
    for item in tree:
        if item["type"] != "blob":
            continue
        ext = os.path.splitext(item["path"])[1]
        if ext not in CODE_EXTENSIONS:
            continue
        content = fetch_file_content(owner, repo, item["path"])
        if content:
            files[item["path"]] = content
    return files


def parse_github_url(url):
    """Parse owner/repo from GitHub URL."""
    url = url.strip().rstrip("/")
    # Handle: https://github.com/owner/repo or github.com/owner/repo
    match = re.match(r"(?:https?://)?github\.com/([^/]+)/([^/]+)", url)
    if match:
        return match.group(1), match.group(2)
    return None, None


# ============ Code Parsing ============

def parse_python_file(filepath, content):
    """Parse a Python file using AST to extract classes, functions, and relationships."""
    nodes = []
    edges = []
    module_name = filepath.replace("/", ".").replace(".py", "")

    # Module node
    nodes.append({
        "id": filepath,
        "type": "module",
        "name": os.path.basename(filepath),
        "file": filepath,
        "description": f"Module {module_name}",
        "code_snippet": content[:500],
    })

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return nodes, edges

    # Imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append({"source": filepath, "target": alias.name, "type": "imports"})
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                edges.append({"source": filepath, "target": node.module, "type": "imports"})
                for alias in node.names:
                    edges.append({"source": filepath, "target": f"{node.module}.{alias.name}", "type": "imports"})

    # Classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_id = f"{filepath}::{node.name}"
            docstring = ast.get_docstring(node) or ""
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            code_lines = content.split("\n")[node.lineno - 1 : node.end_lineno]

            nodes.append({
                "id": class_id,
                "type": "class",
                "name": node.name,
                "file": filepath,
                "line": node.lineno,
                "description": f"Class {node.name}: {docstring[:200]}" if docstring else f"Class {node.name} with methods: {', '.join(methods)}",
                "code_snippet": "\n".join(code_lines[:30]),
                "methods": methods,
            })
            edges.append({"source": filepath, "target": class_id, "type": "defines"})

            # Inheritance
            for base in node.bases:
                if isinstance(base, ast.Name):
                    edges.append({"source": class_id, "target": base.id, "type": "inherits"})
                elif isinstance(base, ast.Attribute):
                    edges.append({"source": class_id, "target": ast.unparse(base), "type": "inherits"})

            # Methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{class_id}.{item.name}"
                    method_doc = ast.get_docstring(item) or ""
                    method_lines = content.split("\n")[item.lineno - 1 : item.end_lineno]

                    nodes.append({
                        "id": method_id,
                        "type": "method",
                        "name": f"{node.name}.{item.name}",
                        "file": filepath,
                        "line": item.lineno,
                        "description": f"Method {node.name}.{item.name}: {method_doc[:200]}" if method_doc else f"Method {node.name}.{item.name}",
                        "code_snippet": "\n".join(method_lines[:20]),
                    })
                    edges.append({"source": class_id, "target": method_id, "type": "has_method"})

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{filepath}::{node.name}"
            docstring = ast.get_docstring(node) or ""
            code_lines = content.split("\n")[node.lineno - 1 : node.end_lineno]

            nodes.append({
                "id": func_id,
                "type": "function",
                "name": node.name,
                "file": filepath,
                "line": node.lineno,
                "description": f"Function {node.name}: {docstring[:200]}" if docstring else f"Function {node.name}",
                "code_snippet": "\n".join(code_lines[:20]),
            })
            edges.append({"source": filepath, "target": func_id, "type": "defines"})

    # Function calls (simple detection)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                edges.append({"source": filepath, "target": node.func.id, "type": "calls"})
            elif isinstance(node.func, ast.Attribute):
                edges.append({"source": filepath, "target": node.func.attr, "type": "calls"})

    return nodes, edges


def parse_generic_file(filepath, content):
    """Basic parsing for non-Python files using regex."""
    nodes = []
    edges = []

    module_name = os.path.basename(filepath)
    nodes.append({
        "id": filepath,
        "type": "module",
        "name": module_name,
        "file": filepath,
        "description": f"Module {module_name}",
        "code_snippet": content[:500],
    })

    # Detect classes
    class_pattern = r'(?:class|interface|struct)\s+(\w+)'
    for match in re.finditer(class_pattern, content):
        class_name = match.group(1)
        class_id = f"{filepath}::{class_name}"
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 300)
        nodes.append({
            "id": class_id,
            "type": "class",
            "name": class_name,
            "file": filepath,
            "description": f"Class {class_name} in {module_name}",
            "code_snippet": content[start:end],
        })
        edges.append({"source": filepath, "target": class_id, "type": "defines"})

    # Detect functions
    func_pattern = r'(?:function|func|def|fn|pub fn|async fn)\s+(\w+)'
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        func_id = f"{filepath}::{func_name}"
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 300)
        nodes.append({
            "id": func_id,
            "type": "function",
            "name": func_name,
            "file": filepath,
            "description": f"Function {func_name} in {module_name}",
            "code_snippet": content[start:end],
        })
        edges.append({"source": filepath, "target": func_id, "type": "defines"})

    # Detect imports
    import_pattern = r'(?:import|from|require|use|include)\s+["\']?([^\s"\';\)]+)'
    for match in re.finditer(import_pattern, content):
        edges.append({"source": filepath, "target": match.group(1), "type": "imports"})

    return nodes, edges


# ============ Graph Building ============

def build_knowledge_graph(files):
    """Parse all files and build a NetworkX knowledge graph."""
    G = nx.DiGraph()
    all_nodes = []

    for filepath, content in files.items():
        if filepath.endswith(".py"):
            nodes, edges = parse_python_file(filepath, content)
        else:
            nodes, edges = parse_generic_file(filepath, content)

        all_nodes.extend(nodes)

        for n in nodes:
            G.add_node(n["id"], **n)

        for e in edges:
            # Only add edges where both nodes exist or target matches a node name
            if G.has_node(e["target"]):
                G.add_edge(e["source"], e["target"], type=e["type"])
            else:
                # Try to resolve target by name matching
                for nid, ndata in G.nodes(data=True):
                    if ndata.get("name") == e["target"]:
                        G.add_edge(e["source"], nid, type=e["type"])
                        break

    return G, all_nodes


# ============ Embeddings & Search ============

@st.cache_resource
def load_embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def build_embedding_index(nodes, embed_model):
    """Build FAISS index from node descriptions."""
    texts = [f"{n['type']}: {n['name']} - {n.get('description', '')}" for n in nodes]
    embeddings = list(embed_model.embed(texts))
    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index


def search_nodes(query, index, nodes, embed_model, top_k=TOP_K):
    """Find the most relevant graph nodes for a query."""
    query_emb = np.array(list(embed_model.embed([query]))).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(nodes):
            results.append({**nodes[idx], "score": float(dist)})
    return results


# ============ Graph Context for LLM ============

def get_subgraph_context(G, node_ids, depth=1):
    """Get a subgraph around the given nodes (ego graph) and format as context."""
    relevant_nodes = set(node_ids)
    for nid in node_ids:
        if G.has_node(nid):
            # Add neighbors up to `depth` hops
            for d in range(1, depth + 1):
                try:
                    ego = nx.ego_graph(G, nid, radius=d, undirected=True)
                    relevant_nodes.update(ego.nodes())
                except nx.NetworkXError:
                    pass

    context_parts = []
    for nid in relevant_nodes:
        if not G.has_node(nid):
            continue
        data = G.nodes[nid]
        part = f"[{data.get('type', 'unknown').upper()}] {data.get('name', nid)}"
        if data.get("file"):
            part += f" (file: {data['file']}"
            if data.get("line"):
                part += f", line {data['line']}"
            part += ")"
        if data.get("description"):
            part += f"\n  Description: {data['description']}"
        if data.get("code_snippet"):
            part += f"\n  Code:\n  ```\n  {data['code_snippet'][:400]}\n  ```"

        # Add relationships
        in_edges = [(u, G.edges[u, nid].get("type", "related")) for u in G.predecessors(nid)]
        out_edges = [(v, G.edges[nid, v].get("type", "related")) for v in G.successors(nid)]
        if in_edges:
            part += f"\n  Incoming: {', '.join(f'{u} --[{t}]-->' for u, t in in_edges[:10])}"
        if out_edges:
            part += f"\n  Outgoing: {', '.join(f'--[{t}]--> {v}' for v, t in out_edges[:10])}"

        context_parts.append(part)

    return "\n\n".join(context_parts), relevant_nodes


# ============ Gemini ============

def ask_gemini(question, graph_context, chat_history):
    """Ask Gemini with graph context."""
    client = genai.Client(api_key=get_key())

    system_instruction = """Bạn là trợ lý AI chuyên phân tích mã nguồn. Bạn được cung cấp thông tin từ knowledge graph của một dự án phần mềm.

Quy tắc:
1. Trả lời dựa trên thông tin từ graph context được cung cấp.
2. Luôn trích dẫn nguồn: ghi [file_path:line] khi đề cập đến code cụ thể.
3. Khi giải thích một class/function, mô tả cả các mối liên hệ (imports, inherits, calls) với các thành phần khác.
4. Nếu thông tin không đủ, nói rõ và gợi ý hướng tìm hiểu thêm.
5. Trả lời bằng tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh."""

    messages = []
    for msg in chat_history:
        messages.append(types.Content(role=msg["role"], parts=[types.Part(text=msg["text"])]))

    user_prompt = f"""Graph context (knowledge graph của dự án):
---
{graph_context}
---

Câu hỏi: {question}"""

    messages.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))

    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text


# ============ Visualization ============

NODE_COLORS = {
    "module": "#4FC3F7",
    "class": "#FF8A65",
    "function": "#81C784",
    "method": "#CE93D8",
}

NODE_SHAPES = {
    "module": "square",
    "class": "diamond",
    "function": "dot",
    "method": "triangle",
}


def visualize_subgraph(G, highlight_nodes, height="500px"):
    """Create an interactive pyvis graph for the given nodes."""
    # Collect subgraph
    sub_nodes = set(highlight_nodes)
    for nid in list(highlight_nodes):
        if G.has_node(nid):
            sub_nodes.update(G.predecessors(nid))
            sub_nodes.update(G.successors(nid))

    subG = G.subgraph(sub_nodes)

    net = Network(height=height, width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for nid in subG.nodes():
        data = G.nodes[nid]
        ntype = data.get("type", "module")
        label = data.get("name", nid)
        title = f"{ntype}: {label}\n{data.get('description', '')[:200]}"
        color = NODE_COLORS.get(ntype, "#90A4AE")
        shape = NODE_SHAPES.get(ntype, "dot")
        size = 30 if nid in highlight_nodes else 18
        border_width = 3 if nid in highlight_nodes else 1

        net.add_node(nid, label=label, title=title, color=color, shape=shape,
                     size=size, borderWidth=border_width, borderWidthSelected=4)

    edge_colors = {
        "imports": "#64B5F6",
        "defines": "#A5D6A7",
        "inherits": "#EF5350",
        "has_method": "#CE93D8",
        "calls": "#FFB74D",
    }

    for u, v in subG.edges():
        etype = G.edges[u, v].get("type", "related")
        net.add_edge(u, v, title=etype, label=etype, color=edge_colors.get(etype, "#78909C"),
                     arrows="to", width=2)

    # Generate HTML
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmpfile.name)
    tmpfile.close()

    with open(tmpfile.name, "r", encoding="utf-8") as f:
        html_content = f.read()
    os.unlink(tmpfile.name)

    return html_content


# ============ Persistence ============

def save_graph_data(G, nodes, index=None):
    """Save graph, nodes, and FAISS index to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    data = nx.node_link_data(G)
    with open(os.path.join(DATA_DIR, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    with open(os.path.join(DATA_DIR, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, default=str)
    if index is not None:
        faiss.write_index(index, INDEX_FILE)


def load_graph_data():
    """Load graph, nodes, and FAISS index from disk."""
    graph_path = os.path.join(DATA_DIR, "graph.json")
    nodes_path = os.path.join(DATA_DIR, "nodes.json")
    if not os.path.exists(graph_path) or not os.path.exists(nodes_path):
        return None, None, None
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=True)
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    index = None
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    return G, nodes, index


def clear_graph_data():
    """Remove saved data."""
    for fname in ["graph.json", "nodes.json", "index.faiss"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            os.remove(path)


# ============ Streamlit UI ============

st.set_page_config(page_title="CodeGraph RAG", page_icon="🔗", layout="wide")
st.title("🔗 CodeGraph RAG")
st.caption("Phân tích mã nguồn GitHub bằng Graph RAG + Gemini")

embed_model = load_embedding_model()

# --- Session state ---
if "graph" not in st.session_state:
    saved_G, saved_nodes, saved_index = load_graph_data()
    if saved_G and saved_nodes:
        st.session_state.graph = saved_G
        st.session_state.graph_nodes = saved_nodes
        st.session_state.faiss_index = saved_index if saved_index else build_embedding_index(saved_nodes, embed_model)
        st.session_state.repo_name = "loaded from disk"
    else:
        st.session_state.graph = None
        st.session_state.graph_nodes = []
        st.session_state.faiss_index = None
        st.session_state.repo_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar ---
with st.sidebar:
    st.header("📦 Repository")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")

    if st.button("🔍 Phân tích", use_container_width=True) and repo_url:
        owner, repo = parse_github_url(repo_url)
        if not owner or not repo:
            st.error("URL không hợp lệ. Ví dụ: https://github.com/owner/repo")
        else:
            with st.spinner(f"Đang fetch mã nguồn {owner}/{repo}..."):
                try:
                    files = fetch_repo_files(owner, repo)
                except Exception as e:
                    st.error(f"Lỗi fetch: {e}")
                    files = {}

            if files:
                st.info(f"Đã fetch {len(files)} file code")

                with st.spinner("Đang phân tích và xây dựng knowledge graph..."):
                    G, all_nodes = build_knowledge_graph(files)
                    idx = build_embedding_index(all_nodes, embed_model)

                    st.session_state.graph = G
                    st.session_state.graph_nodes = all_nodes
                    st.session_state.faiss_index = idx
                    st.session_state.repo_name = f"{owner}/{repo}"
                    st.session_state.messages = []
                    st.session_state.chat_history = []

                    save_graph_data(G, all_nodes, idx)

                st.success(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            else:
                st.error("Không tìm thấy file code nào.")

    # Graph stats
    if st.session_state.graph:
        G = st.session_state.graph
        st.divider()
        st.subheader("📊 Thống kê")
        if st.session_state.repo_name:
            st.write(f"**Repo:** {st.session_state.repo_name}")
        st.write(f"**Nodes:** {G.number_of_nodes()}")
        st.write(f"**Edges:** {G.number_of_edges()}")

        # Count by type
        type_counts = {}
        for _, data in G.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in sorted(type_counts.items()):
            color = NODE_COLORS.get(t, "#ccc")
            st.markdown(f"<span style='color:{color}'>●</span> {t}: **{c}**", unsafe_allow_html=True)

    # Legend
    if st.session_state.graph:
        st.divider()
        st.subheader("🎨 Chú thích")
        for t, color in NODE_COLORS.items():
            st.markdown(f"<span style='color:{color}'>●</span> {t}", unsafe_allow_html=True)

    # Clear
    if st.session_state.graph:
        if st.button("🗑️ Xóa dữ liệu", use_container_width=True):
            st.session_state.graph = None
            st.session_state.graph_nodes = []
            st.session_state.faiss_index = None
            st.session_state.repo_name = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            clear_graph_data()
            st.rerun()


# --- Main ---
if not st.session_state.graph:
    st.info("👈 Paste link GitHub repository ở sidebar để bắt đầu phân tích.")
else:
    # Chat display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "graph_html" in msg:
                components.html(msg["graph_html"], height=520, scrolling=True)

    # Chat input
    if question := st.chat_input("Hỏi về mã nguồn... (vd: 'Giải thích class X', 'Module nào quan trọng nhất?')"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích..."):
                # 1. Semantic search for relevant nodes
                results = search_nodes(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.graph_nodes,
                    embed_model,
                )

                # 2. Get subgraph context
                node_ids = [r["id"] for r in results]
                graph_context, relevant_node_ids = get_subgraph_context(
                    st.session_state.graph, node_ids, depth=1
                )

                # 3. Ask Gemini
                answer = ask_gemini(question, graph_context, st.session_state.chat_history)

            st.markdown(answer)

            # 4. Visualize subgraph
            if relevant_node_ids:
                graph_html = visualize_subgraph(
                    st.session_state.graph,
                    highlight_nodes=set(node_ids[:5]),
                )
                components.html(graph_html, height=520, scrolling=True)

        # Save
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "graph_html": graph_html if relevant_node_ids else None,
        })
        st.session_state.chat_history.append({"role": "user", "text": question})
        st.session_state.chat_history.append({"role": "model", "text": answer})
