# TÀI LIỆU KỸ THUẬT & KIẾN TRÚC HỆ THỐNG RAG (INSIGHT INTO RAG)

Tài liệu này mô tả toàn bộ **công nghệ, luồng dữ liệu và giải thuật** của hệ thống tra cứu tri thức.

Hệ thống vận hành theo mô hình **Adaptive Two-Stage Hybrid RAG**: Query Router thích ứng (Google AI Studio Gemini) → Hybrid Search song song (ChromaDB Dense + BM25 Sparse) → RRF lọc thô → Cross-Encoder lọc tinh (OpenRouter) → LLM tổng hợp (OpenRouter).

---

## 1. TỔNG QUAN TECH STACK

| Thành phần | Công nghệ | Chi tiết |
| :--- | :--- | :--- |
| **Query Transformation** | Google AI Studio (Gemini-2.5-Flash REST API) | Phân luồng thích ứng: Rewrite / Decompose / HyDE / Direct. Fallback tự động về Direct nếu API lỗi. |
| **Dense Vector Index** | ChromaDB + Local BAAI/bge-m3 | 100% Offline via `sentence-transformers`. Singleton, tự nhận CUDA/CPU. 1024 chiều vector. |
| **Sparse Keyword Index** | BM25Okapi + Underthesea | Tách từ tiếng Việt, cache BM25 ra `.pkl` kèm fingerprint SHA-256. |
| **Lọc thô (Stage 1)** | Reciprocal Rank Fusion (RRF) | Gộp Top-15 ChromaDB + Top-15 BM25 → Top-15 ứng viên. Thuần toán học, cực nhanh. |
| **Lọc tinh (Stage 2)** | OpenRouter Cross-Encoder API | Chấm điểm từng cặp [Query, Chunk]. Lọc từ 15 xuống 3-5 đoạn. Fallback Gemini nếu thiếu key. |
| **Generation** | OpenRouter API Gateway | GPT-4o-mini / DeepSeek / Claude / Gemini / Llama... Strict-grounding prompt, trích dẫn nguồn. |
| **Web Server & UI** | FastAPI + Glassmorphism UI | Chat interface + chọn LLM model. Embedding BAAI/bge-m3 cố định (local). |

---

## 2. LUỒNG DỮ LIỆU THỐNG NHẤT (SINGLE PIPELINE)

Toàn bộ hệ thống chỉ có **một luồng RAG duy nhất** — mọi request đều qua cùng một chuỗi xử lý:

```
[ CÂU HỎI NGƯỜI DÙNG ]
         |
         v
[ GIAI ĐOẠN 1: QUERY ROUTER THÍCH ỨNG ]
  Nền tảng: Google AI Studio (Gemini-2.5-Flash Native REST)
  Phân loại tự động:
      |-> rewrite   -> Chuẩn hóa viết tắt, từ lóng ("sổ đỏ" -> "GCNQSDĐ")
      |-> decompose -> Tách câu so sánh/đa vế thành N câu con
      |-> hyde      -> Sinh đoạn văn giả định để search ngữ nghĩa
      +-> direct    -> Truyền thẳng câu hỏi gốc (câu đã rõ ràng)
  * Fallback an toàn: mất mạng / API lỗi -> direct, không gián đoạn
         |
         v
[ GIAI ĐOẠN 2: HYBRID RETRIEVAL SONG SONG ]
    +---------------------------+---------------------------+
    v                                                       v
[ ChromaDB Dense (Top-15) ]             [ BM25 Sparse (Top-15) ]
  BAAI/bge-m3 local, cosine               underthesea word_tokenize
    +---------------------------+---------------------------+
                                |
                                v
[ GIAI ĐOẠN 3: TWO-STAGE FILTERING ]
  Bước 1 - RRF (lọc thô, 0.001s):
      Score(d) = Sum 1/(40 + rank)  ->  Top-15 ứng viên tiềm năng nhất
  Bước 2 - Cross-Encoder Re-ranker (lọc tinh):
      OpenRouter API -> chấm điểm ngữ nghĩa sâu -> Top 3-5 đoạn đắt giá
                                |
                                v
[ GIAI ĐOẠN 4: GENERATION (OpenRouter LLM) ]
  Strict Grounding Prompt -> trả lời dựa đúng 3-5 đoạn ngữ cảnh
  Kèm trích dẫn: tên tài liệu, trang, đoạn văn
```

---

## 3. CẤU TRÚC THƯ MỤC (SAU KHI DỌN DẸP)

```
RAG-tracuu/
├── app.py                          # FastAPI Web Server & API Gateway
├── build_index.py                  # Lập chỉ mục offline (ChromaDB + BM25)
├── main.py                         # CLI Runner (hỏi đáp qua terminal)
├── run_evaluation_50.py            # Đánh giá 50 câu hỏi (pipeline thống nhất)
├── Dockerfile                      # Docker image (CPU torch + pre-download BAAI/bge-m3)
├── docker-compose.yml              # Khởi chạy rag-web / rag-cli
├── requirements.txt                # Python dependencies
├── configs/
│   └── settings.py                 # Cấu hình tập trung (đọc từ .env)
├── data/
│   └── benchmark_testset_50.json   # 50 câu hỏi có Ground Truth
├── chroma_data/                    # ChromaDB index + BM25 cache (sinh ra sau build_index.py)
│   ├── chroma.sqlite3              # Vector index
│   ├── bm25_cache.pkl              # BM25 Okapi index đã tokenize
│   └── documents_cache.pkl         # Cache tài liệu để phục vụ BM25
├── src/
│   ├── models/
│   │   ├── embedding_factory.py    # BAAI/bge-m3 local Singleton
│   │   └── llm_factory.py          # OpenRouter LLM Factory
│   ├── preprocessing/
│   │   ├── document_loader.py      # Nạp PDF/DOCX đa định dạng
│   │   └── legal_chunker.py        # Phân đoạn cấu trúc Chương/Điều/Khoản pháp luật Việt Nam
│   ├── query_transform/
│   │   └── query_router.py         # LLMQueryRouter: Rewrite/Decompose/HyDE/Direct
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # HybridRetriever (RRF -> Cross-Encoder)
│   │   └── reranker.py             # CrossEncoderReranker (OpenRouter + Gemini fallback)
│   ├── storage/
│   │   └── vector_store.py         # VectorDB & HybridVectorDB (ChromaDB + BM25 cache)
│   ├── evaluation/
│   │   ├── ragas_evaluator.py      # 4 Ragas metrics (IR-based thực tế)
│   │   └── visualizer.py           # Biểu đồ PNG/PDF
│   └── pipeline/
│       ├── batch_rag.py            # BatchRAG cốt lõi (retrieve -> generate)
│       └── answer_parser.py        # Parser kết quả LLM
├── static/
│   ├── index.html                  # Giao diện Web
│   ├── css/style.css               # Glassmorphism design
│   └── js/app.js                   # Frontend logic
├── reports/                        # Kết quả đánh giá (CSV, PNG, PDF)
└── ducument/                       # Tài liệu PDF/DOCX nguồn tri thức
```

**Các file đã xoá (không còn tồn tại):**
- `src/pipeline/decomposition_rag.py` — logic tích hợp vào `app.py`
- `src/pipeline/hyde_rag.py` — logic tích hợp vào `app.py`
- `src/query_transform/query_rewriter.py` — chức năng gộp vào `query_router.py`
- `src/query_transform/decomposition.py` — chức năng gộp vào `query_router.py`
- `src/query_transform/hyde.py` — chức năng gộp vào `query_router.py`
- `src/retrieval/bm25_retriever.py` — BM25 chạy trực tiếp trong `vector_store.py`
- `src/preprocessing/pdf_loader.py` — thay bằng `document_loader.py` (hỗ trợ cả DOCX)

---

## 4. CHỈ MỤC & DỮ LIỆU (chroma_data)

**Câu hỏi: Có cần chạy lại `build_index.py --clean` không?**

**KHÔNG** — nếu `chroma_data/chroma.sqlite3` đã tồn tại và có:
- Vectors >= 100
- Sources >= 2 tài liệu nguồn
- Dimension = 1024 (BAAI/bge-m3)

Hệ thống tự phát hiện và nạp trong ~0.01 giây, tuyệt đối không embed lại.

**Chỉ cần chạy lại khi:**
- Thêm tài liệu mới vào `ducument/`
- Xoá hoặc thay đổi tài liệu cũ
- Chỉ mục bị lỗi/hỏng

```bash
# Kiểm tra chỉ mục hiện tại
python -c "from src.storage.vector_store import check_index_health; from configs.settings import settings; h=check_index_health(settings.PERSIST_DIR); print(h)"

# Lập chỉ mục lại từ đầu (chỉ khi cần)
python build_index.py --clean
```

---

## 5. ĐÁNH GIÁ CHẤT LƯỢNG (50 CÂU HỎI)

- **Bộ testset:** `data/benchmark_testset_50.json` — 50 câu, 4 nhóm:
  - `RAG_Theory` (19), `Luật Đất Đai` (13), `Luật Lao Động` (10), `AI_Tech` (8)
- **Phương pháp:** Hỏi từng câu qua pipeline thống nhất (QueryRouter + TwoStageHybridRetriever) → thu thập context → chấm điểm
- **4 chỉ số Ragas (IR-based, thực tế, không hardcode):**
  - `Faithfulness`, `Answer Relevancy`, `Context Precision`, `Context Recall`
- **Báo cáo:** CSV chi tiết + biểu đồ PNG/PDF → `reports/`

```bash
python run_evaluation_50.py              # Toàn bộ 50 câu
python run_evaluation_50.py --sample-size 10  # Thử nhanh 10 câu
```

---

## 6. HƯỚNG DẪN VẬN HÀNH

### 6.1. Local Environment

```powershell
# Lần đầu: lập chỉ mục (bắt buộc)
python build_index.py --clean

# Khởi động Web UI
python app.py
# Truy cập: http://localhost:8000

# CLI hỏi đáp trực tiếp
python main.py --query "Điều kiện cấp sổ đỏ là gì?"
# Hoặc chat tương tác:
python main.py
```

### 6.2. Docker

```bash
# Build image (lần đầu hoặc sau khi thay đổi code/requirements)
docker compose build

# Lập chỉ mục lần đầu (chỉ cần 1 lần, volume persist qua container restart)
docker compose run --rm rag-cli python build_index.py --clean

# Khởi động Web UI
docker compose up -d rag-web
# Truy cập: http://localhost:8000

# Đánh giá 50 câu
docker compose run --rm rag-cli python run_evaluation_50.py

# CLI tương tác
docker compose run --rm rag-cli python main.py
```

> **Lưu ý:** `chroma_data/` được mount làm volume (`./chroma_data:/app/chroma_data`) nên chỉ mục
> tồn tại bền vững kể cả khi restart hay rebuild container.

---

## 7. CẤU HÌNH (.env)

| Biến | Mô tả | Bắt buộc |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | Key cho LLM Generation và Reranker | Có |
| `GEMINI_API_KEY` | Key cho Query Router (Google AI Studio) | Có |
| `OPENROUTER_MODEL` | Model LLM mặc định (vd: `openai/gpt-4o-mini`) | Tùy chọn |
| `RERANKER_MODEL` | Model reranker (vd: `nvidia/llama-nemotron-rerank-vl-1b-v2:free`) | Tùy chọn |
| `EMBEDDING_MODEL` | Luôn là `BAAI/bge-m3` (local, không thay đổi) | - |
| `DATA_DIR` | Thư mục tài liệu nguồn (mặc định: `ducument`) | Tùy chọn |
| `PERSIST_DIR` | Thư mục ChromaDB (mặc định: `chroma_data`) | Tùy chọn |
