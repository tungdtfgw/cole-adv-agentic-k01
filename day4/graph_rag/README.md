# CodeGraph RAG - Phân tích mã nguồn bằng Graph RAG

## Tổng quan

CodeGraph RAG là ứng dụng phân tích mã nguồn từ GitHub sử dụng kỹ thuật **Graph RAG (Graph-based Retrieval-Augmented Generation)**. Thay vì chia code thành các đoạn text rời rạc như RAG truyền thống, Graph RAG xây dựng một **knowledge graph** thể hiện cấu trúc và mối quan hệ giữa các thành phần code (module, class, function), giúp AI hiểu được kiến trúc tổng thể của dự án.

## So sánh RAG truyền thống vs Graph RAG

| | RAG truyền thống | Graph RAG |
|---|---|---|
| **Cấu trúc dữ liệu** | Danh sách chunks phẳng | Đồ thị có nodes + edges |
| **Tìm kiếm** | Tìm đoạn text tương tự | Tìm node + mở rộng theo quan hệ |
| **Context cho AI** | Các đoạn text rời rạc | Subgraph với đầy đủ quan hệ |
| **Phù hợp với** | Tài liệu văn bản | Dữ liệu có cấu trúc quan hệ (code, ontology) |
| **Ví dụ câu hỏi** | "Trang nào nói về X?" | "Class A kế thừa gì, gọi hàm nào, ai import nó?" |

## Kiến trúc hệ thống

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  GitHub URL  │────▶│  Fetch Code  │────▶│  AST / Regex  │────▶│ NetworkX │
│  (user input)│     │  (REST API)  │     │  (parse code) │     │ (DiGraph)│
└──────────────┘     └──────────────┘     └───────────────┘     └────┬─────┘
                                                                     │
                            ┌────────────────────────────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  fastembed   │  Embed mô tả của mỗi node
                     │  + FAISS     │  để tìm kiếm ngữ nghĩa
                     └──────┬───────┘
                            │
┌──────────────┐     ┌──────┴───────┐     ┌───────────────┐     ┌──────────┐
│   PyVis      │◀────│  Subgraph    │◀────│ Semantic      │◀────│ Câu hỏi  │
│ (vẽ đồ thị) │     │  Context     │     │ Search        │     │ user     │
└──────────────┘     └──────┬───────┘     └───────────────┘     └──────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Gemini     │  Sinh câu trả lời
                     │  3.0 Flash   │  từ graph context
                     └──────────────┘
```

**Luồng xử lý gồm 2 pha:**

| Pha | Các bước |
|-----|----------|
| **Indexing** (phân tích repo) | Fetch code → Parse AST → Xây knowledge graph → Embed nodes → Lưu disk |
| **Querying** (hỏi đáp) | Câu hỏi → FAISS tìm nodes → Mở rộng ego graph → Gửi Gemini → Hiển thị + vẽ đồ thị |

## Lựa chọn kỹ thuật

| Thành phần | Thư viện | Lý do chọn |
|------------|----------|-------------|
| **Fetch code** | GitHub REST API (`requests`) | Không cần clone repo, fetch từng file qua HTTP, hỗ trợ mọi public repo |
| **Parse Python** | `ast` (built-in) | Phân tích chính xác 100% cú pháp Python: class, function, import, inheritance, calls |
| **Parse ngôn ngữ khác** | `re` (regex) | Fallback cho JS/TS/Java/Go/... — phát hiện class, function, import bằng pattern matching |
| **Knowledge Graph** | `networkx` (DiGraph) | Thư viện đồ thị chuẩn của Python, hỗ trợ directed graph, ego_graph, subgraph, attributes |
| **Embedding** | `fastembed` (multilingual MiniLM) | Embed mô tả node để tìm kiếm ngữ nghĩa, hỗ trợ đa ngôn ngữ (hỏi tiếng Việt, code tiếng Anh) |
| **Vector search** | `FAISS` (IndexFlatIP) | Tìm nhanh nodes liên quan từ câu hỏi tự nhiên |
| **AI** | Gemini 3.0 Flash | Sinh câu trả lời từ graph context, context window lớn |
| **Visualization** | `pyvis` → `st.components.html` | Render đồ thị tương tác (kéo thả, zoom, hover) trực tiếp trong Streamlit |

### Tại sao cần Embeddings trong Graph RAG?

Embeddings đóng vai trò **cầu nối** giữa câu hỏi ngôn ngữ tự nhiên và các node trong graph:

- Không có embeddings: chỉ tìm được node theo **tên chính xác** (vd: "class Dog")
- Có embeddings: tìm được node theo **ngữ nghĩa** (vd: "xử lý authentication" → tìm ra `AuthMiddleware`, `login()`, `verify_token()`)

Mỗi node được embed dưới dạng: `"{type}: {name} - {description}"`, giúp FAISS hiểu cả loại node lẫn chức năng.

## Cấu trúc Knowledge Graph

### Nodes (đỉnh)

| Loại | Màu | Hình | Mô tả |
|------|------|------|--------|
| `module` | Xanh dương | Vuông | File mã nguồn (.py, .js, ...) |
| `class` | Cam | Kim cương | Class / Interface / Struct |
| `function` | Xanh lá | Tròn | Hàm cấp module |
| `method` | Tím | Tam giác | Method trong class |

Mỗi node chứa metadata: `id`, `name`, `file`, `line`, `description`, `code_snippet`.

### Edges (cạnh)

| Loại | Màu | Ý nghĩa | Ví dụ |
|------|------|---------|-------|
| `defines` | Xanh lá nhạt | Module định nghĩa class/function | `app.py` --defines--> `App` |
| `imports` | Xanh dương nhạt | Module import module/symbol khác | `app.py` --imports--> `flask` |
| `inherits` | Đỏ | Class kế thừa class khác | `Dog` --inherits--> `Animal` |
| `has_method` | Tím | Class chứa method | `Dog` --has_method--> `Dog.speak` |
| `calls` | Cam | Module gọi function/method | `app.py` --calls--> `login` |

## Giải thích các đoạn code quan trọng

### 1. Fetch mã nguồn từ GitHub

```python
# graph_rag.py dòng 30-70
def fetch_github_tree(owner, repo, branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        url = f"...git/trees/master?recursive=1"  # fallback
        resp = requests.get(url, timeout=30)
    return resp.json().get("tree", [])

def fetch_file_content(owner, repo, path, branch="main"):
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(url, timeout=30)
    return resp.text
```

**Cách hoạt động:**
- Gọi GitHub Git Trees API với `?recursive=1` để lấy toàn bộ cây thư mục trong 1 request
- Lọc chỉ lấy file code (`.py`, `.js`, `.ts`, `.java`, `.go`, ...)
- Fetch nội dung từng file qua `raw.githubusercontent.com`
- Tự động fallback từ branch `main` sang `master` nếu cần

### 2. Parse Python bằng AST

```python
# graph_rag.py dòng 85-186
def parse_python_file(filepath, content):
    tree = ast.parse(content)

    # Trích xuất imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.append({"source": filepath, "target": alias.name, "type": "imports"})
        elif isinstance(node, ast.ImportFrom):
            edges.append({"source": filepath, "target": node.module, "type": "imports"})

    # Trích xuất classes + methods
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Tạo node cho class
            # Phát hiện inheritance từ node.bases
            for base in node.bases:
                edges.append({"source": class_id, "target": base.id, "type": "inherits"})
            # Phát hiện methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    edges.append({"source": class_id, "target": method_id, "type": "has_method"})

    # Phát hiện function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            edges.append({"source": filepath, "target": node.func.id, "type": "calls"})
```

**Cách hoạt động:**
- `ast.parse()` biến source code Python thành Abstract Syntax Tree
- `ast.walk()` duyệt toàn bộ tree để tìm imports và function calls
- `ast.iter_child_nodes()` duyệt top-level nodes để tìm class/function definitions
- Với mỗi class, phát hiện: **inheritance** (từ `node.bases`), **methods** (từ `node.body`)
- Mỗi node giữ `code_snippet` (đoạn code gốc) và `description` (docstring hoặc tên + methods)

**Tại sao dùng AST thay vì regex cho Python?**
- AST hiểu đúng cú pháp: nested class, decorator, async function, multiline expressions
- Regex dễ sai với comment, string chứa keyword, code phức tạp
- `ast.get_docstring()` trích xuất docstring chính xác

### 3. Xây dựng Knowledge Graph

```python
# graph_rag.py dòng 248-275
def build_knowledge_graph(files):
    G = nx.DiGraph()

    for filepath, content in files.items():
        if filepath.endswith(".py"):
            nodes, edges = parse_python_file(filepath, content)
        else:
            nodes, edges = parse_generic_file(filepath, content)

        for n in nodes:
            G.add_node(n["id"], **n)

        for e in edges:
            if G.has_node(e["target"]):
                G.add_edge(e["source"], e["target"], type=e["type"])
            else:
                # Resolve by name matching
                for nid, ndata in G.nodes(data=True):
                    if ndata.get("name") == e["target"]:
                        G.add_edge(e["source"], nid, type=e["type"])
                        break
```

**Cách hoạt động:**
- Sử dụng `nx.DiGraph()` — đồ thị **có hướng** (A imports B khác B imports A)
- Mỗi node lưu đầy đủ metadata làm node attributes
- **Edge resolution**: khi target là tên ngắn (vd: `Dog`), tìm node có `name == "Dog"` để nối
- Điều này kết nối được: `file_a.py --calls--> Dog` với `file_b.py::Dog`

### 4. Semantic Search trên Graph

```python
# graph_rag.py dòng 285-307
def build_embedding_index(nodes, embed_model):
    texts = [f"{n['type']}: {n['name']} - {n.get('description', '')}" for n in nodes]
    embeddings = list(embed_model.embed(texts))
    embeddings_np = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index

def search_nodes(query, index, nodes, embed_model, top_k=TOP_K):
    query_emb = np.array(list(embed_model.embed([query]))).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)
    return [nodes[idx] for idx in indices[0] if idx < len(nodes)]
```

**Cách hoạt động:**
- Mỗi node được embed dưới dạng: `"class: Dog - Class Dog with methods: speak, fetch"`
- Format này giúp embedding model hiểu cả **loại** (class/function) lẫn **chức năng**
- Khi user hỏi "xử lý request HTTP", FAISS tìm nodes có mô tả gần nghĩa nhất
- Trả về TOP_K = 10 nodes liên quan nhất

### 5. Mở rộng Subgraph Context (Ego Graph)

```python
# graph_rag.py dòng 312-351
def get_subgraph_context(G, node_ids, depth=1):
    relevant_nodes = set(node_ids)
    for nid in node_ids:
        if G.has_node(nid):
            for d in range(1, depth + 1):
                ego = nx.ego_graph(G, nid, radius=d, undirected=True)
                relevant_nodes.update(ego.nodes())

    # Format context với relationships
    for nid in relevant_nodes:
        data = G.nodes[nid]
        in_edges = [(u, G.edges[u, nid]["type"]) for u in G.predecessors(nid)]
        out_edges = [(v, G.edges[nid, v]["type"]) for v in G.successors(nid)]
```

**Cách hoạt động — đây là phần cốt lõi của Graph RAG:**
1. Từ TOP_K nodes tìm được bởi FAISS, mở rộng ra các **node lân cận** (neighbors)
2. `nx.ego_graph(nid, radius=1)` lấy tất cả nodes cách `nid` tối đa 1 bước (cả incoming và outgoing)
3. Với mỗi node trong subgraph, format đầy đủ: type, name, file, code, **incoming edges**, **outgoing edges**
4. Context gửi cho Gemini chứa **cả cấu trúc quan hệ**, không chỉ text rời rạc

**Ví dụ**: hỏi về `class Dog` → ego graph bao gồm:
- `Dog` (chính)
- `Animal` (Dog inherits Animal)
- `Dog.speak`, `Dog.fetch` (Dog has_method)
- `app.py` (app.py calls Dog)
- `models.py` (models.py defines Dog)

### 6. Trực quan hóa đồ thị

```python
# graph_rag.py dòng 407-456
def visualize_subgraph(G, highlight_nodes, height="500px"):
    sub_nodes = set(highlight_nodes)
    for nid in list(highlight_nodes):
        sub_nodes.update(G.predecessors(nid))
        sub_nodes.update(G.successors(nid))

    subG = G.subgraph(sub_nodes)
    net = Network(height=height, directed=True, bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for nid in subG.nodes():
        net.add_node(nid, label=label, color=color, shape=shape, size=size)
    for u, v in subG.edges():
        net.add_edge(u, v, label=etype, color=edge_color)

    net.save_graph(tmpfile.name)  # → HTML
    components.html(html_content, height=520)  # Render trong Streamlit
```

**Cách hoạt động:**
- Lấy subgraph xung quanh các nodes được highlight (top-5 từ FAISS search)
- `pyvis.Network` tạo đồ thị tương tác dùng vis.js
- `barnes_hut()` thiết lập physics simulation cho layout tự động
- Nodes chính (highlight) có kích thước lớn hơn và viền dày hơn
- Mỗi loại node/edge có **màu và hình dạng riêng** theo bảng chú thích
- Xuất ra HTML, nhúng vào Streamlit qua `st.components.html()`
- User có thể: kéo thả node, zoom, hover xem chi tiết

## Persistence (Lưu trữ)

Dữ liệu được lưu tự động vào thư mục `graph_data/`:

| File | Nội dung |
|------|----------|
| `graph.json` | Knowledge graph (nodes + edges) dạng node-link JSON |
| `nodes.json` | Danh sách nodes với metadata đầy đủ |

Khi khởi động lại server, hệ thống tự động load graph từ disk và rebuild FAISS index.

## Chạy ứng dụng

```bash
cd day4/graph_rag
streamlit run graph_rag.py
```

## Ví dụ sử dụng

1. Paste URL: `https://github.com/pallets/flask`
2. Bấm "Phân tích" → hệ thống fetch code, xây graph
3. Hỏi: "Giải thích class Flask" → Gemini trả lời + đồ thị hiển thị Flask và các class liên quan
4. Hỏi: "Module nào xử lý routing?" → tìm các nodes liên quan đến routing, vẽ mối quan hệ
5. Hỏi: "Class nào kế thừa từ BaseResponse?" → truy vấn quan hệ `inherits` trong graph
