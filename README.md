# Project: Insight into RAG (AI VIET NAM Course 2026)

Hệ thống tra cứu tri thức thông minh với **Adaptive Two-Stage Hybrid RAG**: Query Router thích ứng (Google Gemini) → Hybrid Search (ChromaDB + BM25) → RRF → Cross-Encoder → LLM.

---

## Tính Năng Chính

- **Một luồng RAG thống nhất:** QueryRouter tự động chọn chiến lược tối ưu (Rewrite / Decompose / HyDE / Direct) cho từng câu hỏi.
- **Embedding Local 100% offline:** BAAI/bge-m3 qua `sentence-transformers`, không tốn API, không rate limit, 1024 chiều.
- **Hybrid Search + Two-Stage Filter:** BM25 + ChromaDB → RRF (lọc thô) → Cross-Encoder (lọc tinh) → 3-5 đoạn chất lượng nhất.
- **LLM đa dạng qua OpenRouter:** GPT-4o-mini, DeepSeek, Claude, Gemini, Llama... đổi model bằng 1 biến môi trường.
- **Đánh giá 50 câu hỏi thực tế:** Faithfulness, Answer Relevancy, Context Precision, Context Recall — không hardcode.

---

## Bắt Đầu Nhanh

### 1. Cài đặt & Cấu hình

```bash
pip install -r requirements.txt
# Tạo file .env với các key cần thiết:
# OPENROUTER_API_KEY=sk-or-v1-xxx
# GEMINI_API_KEY=AIza-xxx
```

### 2. Lập chỉ mục tài liệu (chỉ cần 1 lần)

```bash
python build_index.py --clean
```

> Đọc toàn bộ PDF/DOCX trong `ducument/`, tạo ChromaDB + BM25 cache vào `chroma_data/`.  
> Khi đã có chỉ mục, hệ thống nạp trong **~0.01 giây**, tuyệt đối không embed lại.

### 3. Khởi động Web UI

```bash
python app.py
```

Truy cập: **http://localhost:8000** — Chat interface + chọn LLM model.

### 4. CLI hỏi đáp trực tiếp

```bash
# Single question
python main.py --query "Điều kiện cấp sổ đỏ là gì?"

# Chat tương tác
python main.py
```

### 5. Đánh giá 50 câu hỏi Benchmark

```bash
# Toàn bộ 50 câu
python run_evaluation_50.py

# Thử nhanh 10 câu
python run_evaluation_50.py --sample-size 10
```

Kết quả xuất ra `reports/` (CSV + biểu đồ PNG/PDF).

---

## Chạy Bằng Docker

```bash
# Build image (lần đầu)
docker compose build

# Lập chỉ mục (1 lần, persist qua volume)
docker compose run --rm rag-cli python build_index.py --clean

# Khởi động Web UI
docker compose up -d rag-web
# Truy cập: http://localhost:8000

# Đánh giá 50 câu
docker compose run --rm rag-cli python run_evaluation_50.py

# CLI hỏi đáp
docker compose run --rm rag-cli python main.py --query "RAG là gì?"
```

---

## Cấu Hình (.env)

| Biến | Mô tả | Bắt buộc |
|:---|:---|:---:|
| `OPENROUTER_API_KEY` | Key cho LLM & Reranker | ✅ |
| `GEMINI_API_KEY` | Key cho Query Router (Google AI Studio) | ✅ |
| `OPENROUTER_MODEL` | Model LLM (mặc định: `openai/gpt-4o-mini`) | - |
| `RERANKER_MODEL` | Reranker (mặc định: `nvidia/llama-nemotron-rerank-vl-1b-v2:free`) | - |
| `DATA_DIR` | Thư mục tài liệu (mặc định: `ducument`) | - |
| `PERSIST_DIR` | ChromaDB index (mặc định: `chroma_data`) | - |

---

## Tài Liệu Kỹ Thuật

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Kiến trúc hệ thống, luồng dữ liệu, cấu trúc thư mục
- [review.md](review.md) — Lịch sử rà soát & thay đổi codebase
