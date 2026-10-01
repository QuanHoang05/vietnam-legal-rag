# ⚖️ Vietnam Legal RAG

> **Hệ thống Tra cứu & Hỏi đáp Pháp luật Việt Nam Thông minh sử dụng kiến trúc Adaptive Two-Stage Hybrid RAG (Pháp điển hóa Luật Đất đai 2024 & Bộ luật Lao động 2019)**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Dense%20Vector-orange.svg)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/Model-AITeamVN%2FVietnamese__Reranker-yellow.svg)](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
[![Ragas Evaluation](https://img.shields.io/badge/Evaluation-Ragas%20Official-green.svg)](https://docs.ragas.io/)

---

## 📌 1. Giới thiệu Tổng quan

**Vietnam Legal RAG** là giải pháp tra cứu và hỏi đáp ngữ nghĩa chuyên sâu dành riêng cho hệ thống văn bản quy phạm pháp luật Việt Nam. Dự án giải quyết triệt để các hạn chế cố hữu của các hệ thống RAG thông thường (văn bản luật dài, đứt đoạn ngữ cảnh Điều/Khoản, câu hỏi viết tắt, khẩu ngữ đời thường và hiện tượng ảo giác thông tin của LLM).

Hệ thống được xây dựng trên một **Pipeline Thống nhất (Single Unified Pipeline)**, kết hợp giữa mô hình phân loại truy vấn thông minh, tìm kiếm lai đa tầng (Dense + Sparse) và mô hình tái xếp hạng Cross-Encoder tiếng Việt chạy cục bộ 100% offline.

---

## 🏗️ 2. Kiến trúc Hệ thống (Adaptive Two-Stage Hybrid RAG)

```
[Người dùng đặt câu hỏi]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 1. ADAPTIVE QUERY ROUTER (Google AI Studio / Gemini)   │
│ Phân loại chiến lược: Direct / Rewrite / Decompose/HyDE│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. STAGE 1: HYBRID RETRIEVAL & RECIPROCAL RANK FUSION │
│  • Dense Vector: ChromaDB (BAAI/bge-m3, 1024 dims)    │
│  • Sparse Keyword: BM25Okapi (Tiếng Việt)              │
│  • Thuật toán RRF (k=40) ──> Lọc thô Top 15 ứng viên   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. STAGE 2: LOCAL CROSS-ENCODER RERANKER               │
│ Mô hình: AITeamVN/Vietnamese_Reranker (PyTorch local)  │
│ Chấm điểm tương quan ngữ nghĩa sâu ──> Top 5 đoạn tốt   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. STRICT GENERATION & PARSER (LLM + Anti-Hallucination)│
│  • Mô hình: Gemini-2.5-Flash / OpenRouter Auto-Cascade │
│  • Prompting nghiêm ngặt (Chỉ dựa trên điều luật trích)│
│  • FocusedAnswerParser (Làm sạch kết quả)              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
               [Câu trả lời chuẩn xác & Trích dẫn Điều luật]
```

---

## 🚀 3. Điểm nổi bật & Cải tiến Kỹ thuật

### 🔹 Phân đoạn Cấu trúc Pháp lý (Legal Article Chunking)
- **Tôn trọng đơn vị ngữ nghĩa nguyên vẹn**: Không chia văn bản theo số lượng ký tự ngẫu nhiên. Mỗi **Điều luật** được coi là một khối độc lập (Atomic Semantic Unit).
- **Chia nhỏ có điều kiện (Conditional Splitting)**: Chỉ phân rã thành từng Khoản khi Điều luật quá dài (>1400 ký tự).
- **Tiêm ngữ cảnh phân cấp (Breadcrumb Context Injection)**: Mọi chunk con đều được gắn tiền tố nhận diện: `[Tên Luật | Tên Chương | Điều X. Tiêu đề (Khoản Y)]`.
- **Định danh SHA-256 xác định**: Ngăn ngừa trùng lặp dữ liệu trong Vector Database.

### 🔹 Bộ định tuyến Truy vấn Thích ứng (Adaptive Query Router)
Tự động phân tích câu hỏi người dùng thành 4 chiến lược:
1. `direct`: Câu hỏi đã rõ ràng, thuật ngữ chuẩn $\rightarrow$ Truy vấn trực tiếp.
2. `rewrite`: Câu hỏi dùng khẩu ngữ (*"sổ đỏ"*, *"quỵt lương"*, *"đuổi việc"*) hoặc viết tắt (*"đđ"*, *"bhxh"*, *"hđlđ"*) $\rightarrow$ Viết lại thành thuật ngữ pháp lý chính quy.
3. `decompose`: Câu hỏi phức tạp, so sánh đối chiếu $\rightarrow$ Phân rã thành 2-3 câu hỏi con độc lập.
4. `hyde`: Câu hỏi trừu tượng, giải thích cơ chế $\rightarrow$ Tạo tài liệu giả định mang văn phong quy phạm pháp luật.
- **Dự phòng Đa tầng (Cascade Fallback)**: Nếu Google Gemini chạm hạn mức (HTTP 429), hệ thống tự động fallback sang OpenRouter (`gpt-4o-mini`).

### 🔹 Tái xếp hạng Cục bộ Tiếng Việt (Local Vietnamese Reranker)
- Tích hợp mô hình Cross-Encoder chuyên sâu **`AITeamVN/Vietnamese_Reranker`**.
- Chạy trực tiếp trên máy cục bộ (CPU/GPU) thông qua PyTorch & Transformers, **không tốn chi phí API, 100% offline, không phụ thuộc mạng**.
- Tối ưu `max_length=512` giúp tốc độ suy luận nhanh gấp 4 lần trên CPU thông thường.

### 🔹 Quản trị Cấu hình Tập trung 100% (Centralized Settings)
- Toàn bộ tham số hệ thống (API keys, URLs, model selection, timeouts, candidate_k, top_k, chunk size, rate limit, host, port) được quản lý thống nhất tại [`configs/settings.py`](configs/settings.py).
- Hỗ trợ ghi đè linh hoạt qua file [`.env`](.env) mà không cần can thiệp mã nguồn.

---

## 📂 4. Cấu trúc Thư mục Dự án

```plaintext
vietnam-legal-rag/
├── configs/
│   └── settings.py              # Trung tâm cấu hình toàn bộ hệ thống
├── data/
│   └── benchmark_testset_50.json# Bộ câu hỏi kiểm thử chuẩn hóa 50 câu
├── ducument/                    # Thư mục chứa tài liệu PDF/DOCX (Luật Đất đai, Luật Lao động)
├── src/
│   ├── models/
│   │   ├── embedding_factory.py # Embedding BAAI/bge-m3 cục bộ
│   │   └── llm_factory.py       # CascadeChatModel (Gemini + OpenRouter)
│   ├── preprocessing/
│   │   ├── document_loader.py   # Bộ nạp PDF/Word đa định dạng
│   │   ├── legal_chunker.py     # Bộ chia đoạn phân cấp Chương/Điều/Khoản
│   │   └── text_cleaner.py      # Chuẩn hóa tiếng Việt Unicode NFC
│   ├── query_transform/
│   │   └── query_router.py      # LLM Query Router & Rewriter thích ứng
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Two-Stage Hybrid (BM25 + Chroma + RRF)
│   │   └── reranker.py          # Local Vietnamese Cross-Encoder Reranker
│   ├── storage/
│   │   └── vector_store.py      # ChromaDB & Pickle BM25 Cache
│   ├── pipeline/
│   │   ├── batch_rag.py         # Điều phối luồng xử lý RAG & Prompting
│   │   └── answer_parser.py     # Hậu xử lý và bóc tách câu trả lời
│   └── evaluation/
│       ├── ragas_evaluator.py   # Chấm điểm 4 metric Ragas chính thức
│       └── visualizer.py        # Vẽ biểu đồ kết quả PNG / PDF
├── static/                      # Giao diện Web SPA (HTML, CSS, JS cao cấp)
├── app.py                       # Máy chủ FastAPI Web Server & REST API
├── build_index.py               # Script lập chỉ mục tri thức ngoại tuyến
├── main.py                      # Giao diện dòng lệnh tra cứu CLI
├── run_evaluation_50.py         # Script chạy benchmark 50 câu hỏi Ragas
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── .env.example                 # File mẫu cấu hình biến môi trường
└── README.md                    # Tài liệu hướng dẫn dự án
```

---

## 🛠️ 5. Hướng dẫn Cài đặt & Sử dụng

### Bước 1: Clone kho mã nguồn
```bash
git clone https://github.com/QuanHoang05/vietnam-legal-rag.git
cd vietnam-legal-rag
```

### Bước 2: Thiết lập môi trường Python
Khuyến nghị sử dụng Python 3.10 (Conda hoặc Virtualenv):
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows:
.\venv\Scripts\activate

# Hoặc kích hoạt trên Linux/macOS:
source venv/bin/activate

# Cài đặt các thư viện phụ thuộc:
pip install -r requirements.txt
```

### Bước 3: Cấu hình Biến môi trường
Sao chép file `.env.example` thành `.env` và cập nhật API Key:
```bash
cp .env.example .env
```
Mở file `.env` và điền:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Bước 4: Lập chỉ mục Kho tri thức (Offline Ingestion)
Chạy script phân đoạn và lập chỉ mục Vector DB (chỉ cần chạy 1 lần duy nhất):
```bash
python build_index.py --clean
```

---

## 💻 6. Thực thi Hệ thống

### 🌐 1. Khởi chạy Giao diện Web (FastAPI)
```bash
python app.py
```
- Mở trình duyệt truy cập: **`http://localhost:8000`**
- Tài liệu API tương tác Swagger UI: **`http://localhost:8000/docs`**

### ⌨️ 2. Tra cứu bằng Dòng lệnh (CLI)
```bash
python main.py
```
Hoặc chỉ định mô hình tùy chọn:
```bash
python main.py --model openai/gpt-4o-mini
```

### 📊 3. Chạy Đánh giá Benchmark 50 Câu (Ragas Evaluation)
Chạy toàn bộ quy trình kiểm thử và chấm điểm tự động 50 câu benchmark:
```bash
python run_evaluation_50.py
```
- Kết quả chi tiết xuất ra thư mục `reports/`:
  - `reports/benchmark_report_<timestamp>.csv`
  - `reports/benchmark_report_<timestamp>.png`

---

## 📈 7. Tiêu chuẩn Đánh giá Benchmark (Ragas Metrics)

Hệ thống được đánh giá khách quan dựa trên 4 chỉ số cốt lõi:
1. **Faithfulness (Độ trung thực)**: Đảm bảo câu trả lời hoàn toàn bắt nguồn từ văn bản luật trích xuất, 0% bịa đặt.
2. **Answer Relevancy (Độ phù hợp của câu trả lời)**: Trả lời đúng trọng tâm câu hỏi của người dùng.
3. **Context Precision (Độ chính xác ngữ cảnh)**: Đo lường mức độ ưu tiên của các Điều luật quan trọng được đẩy lên đầu nhờ Cross-Encoder Reranker.
4. **Context Recall (Độ bao phủ ngữ cảnh)**: Đảm bảo không bỏ sót các Điều/Khoản cần thiết để giải quyết câu hỏi.

---

## 📄 8. Giấy phép (License)
Dự án được phát hành dưới giấy phép [MIT License](LICENSE).
