# REVIEW — RÀ SOÁT & SỬA ĐỔI CODEBASE (2026-09-29)

## TÓM TẮT

---

## 1. `docs/ARCHITECTURE.md`

**Đã làm (lần 1):** Viết lại hoàn toàn — bỏ mô tả 9-mode cũ, mô tả đúng 1 luồng thống nhất.

**Đã làm (lần 2 — sau dọn code):** Cập nhật cấu trúc thư mục phản ánh đúng files thực tế sau khi xoá, bổ sung:
- Mục "Các file đã xoá" ghi rõ lý do xoá
- Mục 4 giải thích khi nào cần/không cần chạy lại `build_index.py`
- Hướng dẫn Docker cập nhật với `docker compose build` trước khi dùng

---

## 2. `run_evaluation_50.py`

**Vấn đề:** Mode selector cũ, không dùng QueryRouter hay TwoStageHybridRetriever → kết quả đánh giá sai pipeline.

**Đã sửa:** Viết lại hoàn toàn — 1 luồng duy nhất, QueryRouter + TwoStageHybridRetriever, in tiến độ `[01/50]`, lưu cột `strategy` vào CSV.

---

## 3. `main.py`

**Vấn đề cũ (lỗi crash):**
- `from src.preprocessing.pdf_loader import SimpleLoader` — file không tồn tại
- `args.data_dir` dùng nhưng không có `--data-dir` trong parser
- `--benchmark` gọi `experiments/run_all_benchmarks.py` không tồn tại

**Đã sửa:** Viết lại — 1 luồng pipeline, xoá 9-mode selector, xoá args lỗi.

---

## 4. `requirements.txt`

**Đã thêm:**
```
python-docx>=1.1.0    # document_loader.py load DOCX nhưng thiếu dep
ragas>=0.1.0          # ragas_evaluator.py import nhưng không khai báo
datasets>=2.18.0      # ragas cần datasets
```

---

## 5. `docker-compose.yml`

**Vấn đề cũ:**
- `rag-benchmark` chạy `experiments/run_all_benchmarks.py` — không tồn tại → container fail
- `rag-cli` thiếu mount `.env`, `build_index.py`, `run_evaluation_50.py`

**Đã sửa:** Xoá `rag-benchmark`, fix volumes cho `rag-cli`, thêm `restart: unless-stopped` cho `rag-web`, dùng chung image.

---

## 6. `Dockerfile`

**Đã thêm:** `ENV HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, bước pre-download BAAI/bge-m3 lúc build.

---

## 7. DỌN CODE THỪA (lần 2)

### Files đã xoá:

| File | Lý do |
|:---|:---|
| `src/pipeline/decomposition_rag.py` | Logic đã tích hợp vào `app.py` (QueryRouter strategy=decompose) |
| `src/pipeline/hyde_rag.py` | Logic đã tích hợp vào `app.py` (QueryRouter strategy=hyde) |
| `src/query_transform/query_rewriter.py` | Chức năng tích hợp vào `query_router.py` (strategy=rewrite) |
| `src/query_transform/decomposition.py` | Chức năng tích hợp vào `query_router.py` (strategy=decompose) |
| `src/query_transform/hyde.py` | Chức năng tích hợp vào `query_router.py` (strategy=hyde) |
| `src/retrieval/bm25_retriever.py` | BM25 chạy trực tiếp trong `vector_store.py` via `BM25Okapi` |
| `src/preprocessing/pdf_loader.py` | Thay bằng `document_loader.py` (load cả PDF + DOCX) |

### Code thừa đã xoá trong các file còn lại:

| File | Xoá gì |
|:---|:---|
| `app.py` | Import `HyDEBatchRAG`, `DecompositionBatchRAG`, `BM25Retriever`, `VectorDB` |
| `app.py` | `PipelineManager`: xoá `vector_dbs`, `hybrid_dbs`, `semantic_dbs`, `split_docs_*`, `bm25_retriever`, `get_vector_db()`, `get_semantic_db()` |
| `app.py` | Các mode riêng: `bm25`, `hyde`, `decomposition`, `rerank`, `mmr`, `semantic`, `baseline` — tất cả chuyển vào luồng unified |
| `src/retrieval/hybrid_retriever.py` | Xoá `InterleavingHybridRetriever` (class không dùng ở đâu) |
| `src/retrieval/__init__.py` | Xoá import `BM25Retriever`, `InterleavingHybridRetriever` |
| `src/pipeline/__init__.py` | Xoá import `HyDEBatchRAG`, `DecompositionBatchRAG` |
| `src/query_transform/__init__.py` | Xoá import `HyDEQueryTransformer`, `DecompositionQueryTransformer`, `QueryRewriter` |

---

## 8. VỀ CHỈ MỤC (chroma_data)

**Tình trạng hiện tại (đã kiểm tra):**
```
Vectors=4795  Sources=12  Dimension=1024  STATUS: HEALTHY
```

**Kết luận: KHÔNG CẦN chạy lại `build_index.py --clean`.**

Chỉ mục đang hoàn toàn hợp lệ với 4,795 vectors từ 12 nguồn tài liệu, chiều 1024 (đúng BAAI/bge-m3). Hệ thống sẽ nạp trực tiếp trong ~0.01 giây khi khởi động.

**Chỉ cần chạy lại khi:** thêm/sửa/xoá tài liệu trong `ducument/`.

---

## 9. CẤU TRÚC THƯ MỤC SAU DỌN DẸP

```
src/
├── models/
│   ├── embedding_factory.py      # BAAI/bge-m3 Singleton
│   └── llm_factory.py            # OpenRouter LLM
├── preprocessing/
│   ├── document_loader.py        # PDF + DOCX loader
│   ├── text_cleaner.py           # Unicode NFC
│   ├── hierarchical_chunker.py   # Parent-Child chunking
│   └── chunking.py               # Recursive & Semantic chunking
├── query_transform/
│   └── query_router.py           # LLMQueryRouter (Gemini REST)
├── retrieval/
│   ├── hybrid_retriever.py       # HybridRetriever (RRF + Reranker)
│   └── reranker.py               # CrossEncoderReranker
├── storage/
│   └── vector_store.py           # VectorDB + HybridVectorDB
├── evaluation/
│   ├── ragas_evaluator.py        # 4 metrics IR-based
│   └── visualizer.py             # Charts
└── pipeline/
    ├── batch_rag.py              # BatchRAG core
    └── answer_parser.py          # LLM output parser
```

---

# REVIEW LẦN 2 — RÀ SOÁT TOÀN DIỆN (2026-10-01)

> **Phạm vi:** 26 file `.py` (~2.971 LOC) + frontend 1.402 LOC + Docker.
> **Căn cứ:** đọc toàn bộ mã nguồn, truy vấn SQL trực tiếp `chroma_data/chroma.sqlite3`,
> đo đạc trên corpus thật 4.795 chunk, và live call có kiểm chứng.
> **Lưu ý:** `ragas_evaluator.py` được viết lại lúc 07:27, benchmark chạy lúc 07:36 cùng ngày.
> Báo cáo này phản ánh trạng thái **sau 07:36** — khác nhiều so với bản review 29/09.

## TÓM TẮT

> **Kiến trúc đúng. Kỹ thuật chọn lọc tốt. Nhưng hệ thống trên đĩa KHÔNG làm những gì nó tự nhận.**

7 lỗi mức đỏ, tất cả đều là **degrade âm thầm** — chạy tiếp bình thường, không lỗi, không log,
nhưng kết quả sai. Đây là loại bug tệ nhất vì nó làm hỏng kết quả mà không ai thấy.

| Hạng mục | Điểm | Ghi chú |
| :--- | :---: | :--- |
| Phân tầng module | 4/5 | Đúng hướng, không import vòng |
| Quy ước đặt tên | 4/5 | Nhất quán |
| Kích thước file | 3/5 | `app.py` 482 dòng gộp nhiều vai trò |
| DRY | 2/5 | Pipeline dựng 3 bản, đã lệch nhau |
| Quản lý cấu hình | 2/5 | 11/13 knob chết |
| **Quản lý phiên bản** | **0/5** | **Không phải git repo** |
| **Kiểm thử** | **0/5** | **0 test, 0 CI** |

---

## 1. CẤU TRÚC

### 1.1 Vấn đề cấu trúc

**S1 — Không có git.** `git rev-parse` → `not a git repository`.
`.gitignore` viết cẩn thận nhưng vô tác dụng. Với dự án CV, đây là điểm trừ lớn nhất:
không có lịch sử để chứng minh tiến bộ, không có repo để link.

**S2 — Pipeline dựng 3 lần, đã lệch nhau.**

| Tham số | `app.py:309-315` | `main.py:59-71` | `run_evaluation_50.py:145-157` |
| :--- | :--- | :--- | :--- |
| `candidate_k` | 15 | 15 | 15 |
| `k` | `req.top_k or 5` | 5 | 5 |
| reranker `top_k` | `req.top_k or 5` | 5 | 5 |

Dispatcher 4 nhánh (direct/rewrite/decompose/hyde) bị copy nguyên văn ở
`app.py:315-375` và `main.py:87-124`. **Hệ quả đã ăn thật:** bản `main.py` không được
cập nhật và crash — xem lỗi F2.

**S3 — 11/13 knob cấu hình chết.** Không có call-site nào ngoài `settings.py`:
`EMBEDDING_PROVIDER`, `BM25_K`, `HYBRID_K`, `HYDE_K`, `HYBRID_INTERLEAVING_TOP_K`,
`SEMANTIC_*`, `MIN_CHUNK_SIZE`, `DECOMPOSITION_MAX_SUB_QUESTIONS`, `GEMINI_BASE_URL`,
`RETRIEVE_K`, `RERANK_TOP_K`.

> ⚠️ Nguy hiểm nhất: `ASSESSMENT_live_eval_50.md:169` khuyên *"Tăng `HYBRID_K` lên 12-15"*.
> **Làm theo sẽ không có gì thay đổi** — `candidate_k`/`k` đều hardcode ở cả 3 call-site.

**S4 — Alias trùng tên.** `TwoStageHybridRetriever = HybridRetriever` (`hybrid_retriever.py:107`),
`OpenRouterReranker = CrossEncoderReranker` (`reranker.py:142`).
Docs viết một tên, code định nghĩa tên kia.

**S5 — Doc lệch code.**
- `ARCHITECTURE.md:90` liệt kê `chunking.py` — file không tồn tại (chỉ còn `.pyc` cũ).
- Docs ghi "fingerprint SHA-256" nhưng `vector_store.py:25` cắt còn `hexdigest()[:16]` = 64 bit.

---

## 2. CHỨC NĂNG

### F1. 🔴 Lập chỉ mục chỉ gom 12,6% corpus (nghiêm trọng nhất toàn hệ thống)

`document_loader.py:23` đặt `max_pages = 50`, không chỗ nào override.
Đo trực tiếp SQL trên `chroma.sqlite3`:

| File | Trang trên đĩa | Đã index | % |
| :--- | ---: | ---: | ---: |
| `AIO_Exercise_Book_v2025 (1).pdf` | 2.028 | 50 | **2,5%** |
| `LLM.pdf` | 603 | 50 | **8,3%** |
| `AIO2025_Reading.pdf` | 247 | 50 | 20,2% |
| `[Reading]-RAG-System.pdf` | 113 | 50 | 44,2% |
| **Tổng PDF** | **3.339** | **421** | **12,6%** |

DOCX không giới hạn trang nên chiếm 2.527/4.795 chunk = **53% chỉ mục là văn bản luật**.
Đây chính là nguyên nhân `RAG_Theory` và `AI_Tech` điểm thấp nhất — tài liệu chứa kiến thức
đó đã bị cắt. Không có cảnh báo nào; `check_index_health` vẫn trả `healthy`.

**Ba lỗi lập chỉ mục đi kèm:**

- `vector_store.py:100` `if not documents or count >= len(documents)` → **discard tài liệu mới,
  giữ index cũ, vẫn báo thành công.** Đã tái hiện: index 3 chunk, rebuild với 2 chunk đúng
  → ghi 0 dòng, trả về nội dung cũ.
- `vector_store.py:116/204` gọi `from_documents` không `delete_collection` → rebuild không
  `--clean` **cộng dồn orphan**, retrieval trả về chunk của file đã xoá.
- `_get_doc_fingerprint` chỉ hash `len(docs)` + mẫu 51 doc + `chunk_id[:8]` → sửa nội dung
  doc thứ 137/500 **fingerprint không đổi**, BM25 cache cũ được dùng lại.

### F2. 🔴 CLI chết ở 2/4 chiến lược

`batch_rag.py:34-40` template có `{history_block}`.
`main.py:103` và `main.py:111` gọi `.format(context=..., question=...)` — **thiếu
`history_block`** → `KeyError`. `app.py:342` truyền đủ 3 tham số.
Tức `main.py --query` chết với `decompose` và `hyde`.

### F3. 🔴 Query Router là no-op — Gemini quota 20 request/ngày

Live call với key thật trong `.env`:

```
HTTP 429 — quotaId: GenerateRequestsPerDayPerModel-FreeTier, quotaValue: "20"
```

Bằng chứng trong chính báo cáo của dự án: cột `strategy` trong
`eval_50_unified_pipeline.csv` = **`direct` cho 50/50 dòng**.

`query_router.py:108-116` nuốt mọi lỗi rồi trả về `direct` — **trông như quyết định thật**.
Người đọc CSV không bao giờ biết router đã chết.
`README.md:9` claim *"QueryRouter tự động chọn chiến lược tối ưu"* — số liệu của chính dự án
phủ nhận điều đó.

Hệ quả trực tiếp: đúng nhóm query viết tắt mà router sinh ra để sửa
(`đđ`, `bhxh`, `sa thải`) giờ đi thẳng vào BM25 — xem F4.

### F4. 🔴 RAG bị nhiễu bởi BM25 score = 0

`hybrid_retriever.py:83-86` không lọc `score > 0`.
Đo trên corpus thật 4.795 chunk:

| Query | Dòng khớp thật | Candidate score = 0 |
| :--- | ---: | ---: |
| `đđ` | 0 | **15/15** |
| `bhxh` | 0 | **15/15** |
| `sa thải` | 6 | 9/15 |
| `nghỉ phép năm` | 319 | 0/15 ✅ |

`np.argsort` trên vector toàn 0 trả `[19 18 17 16 15]` — tức **15 chunk tùy ý đẩy vào pool RRF**.

### F5. 🔴 Reranker chết âm thầm, không một dòng log

`reranker.py:56-72` **không có nhánh `else` cho HTTP non-200** →
401/402/404/429 rơi thẳng xuống `return None` in gì cả.

Live: `X-RateLimit-Limit: 50, Remaining: 0` (model `...:free`).
`rerank()` trả `documents[:top_k]` — **giống hệt input**, nên
`hybrid_retriever.py:96` `if reranked_docs:` không phân biệt được
"đã rerank" với "chưa rerank gì". Không ai biết tầng Cross-Encoder đã tắt.

Thêm: `decompose` gọi `retriever.invoke()` cho từng sub-query → **5 call rerank/câu hỏi**
→ 10 câu là hết quota ngày.

### F6. 🔴 Điểm Ragas đo trên 3–14 mẫu, không phải 50

> **Thay đổi tốt:** heuristic đã bị xoá, `evaluate_ragas` **giờ chạy Ragas thật**.

Nhưng số mẫu chấm được rất thấp:

| Metric | Số mẫu chấm | Mean |
| :--- | ---: | ---: |
| Faithfulness | **3/50** | 0.6583 |
| Answer Relevancy | **9/50** | 0.8412 |
| Context Precision | **3/50** | 0.6292 |
| Context Recall | **14/50** | 0.6369 |

**Nguyên nhân:** `EVAL_MAX_WORKERS = 16` (`settings.py:112`) đè lên OpenRouter bị giới hạn
→ `raise_exceptions=False` (`:120`) biến mỗi lỗi thành `NaN`
→ `dropna().mean()` (`:132`) âm thầm loại 70–94% mẫu.

Đã chứng minh ngược lại: chạy lại 2 câu đó với `max_workers=2` cho
`context_recall 0.667 / 1.0` và `faithfulness 1.0 / 1.0`.
**Metrics chạy tốt, chỉ là chạy quá tay.**

Tệ hơn: hệ thống càng hỏng thì điểm càng cao — câu lỗi bị loại khỏi trung bình.

### F7. 🔴 `/api/benchmark` 404 vĩnh viễn

`app.py:433` đọc `reports/benchmark_summary.csv`,
nhưng `run_evaluation_50.py:201` ghi `eval_50_unified_pipeline.csv`. Sai tên file.

---

## 3. VẤN ĐỀ MỨC VỪA (chọn lọc)

| # | Vấn đề | Vị trí |
| :--: | :--- | :--- |
| M1 | **Gemini API key lọt vào log** — key nằm trong query string URL, exception bị `print` ra, urllib3 nhúng URL vào message | `query_router.py:47,83` |
| M2 | 500 trả nguyên text exception; host `0.0.0.0`; không auth | `app.py:382,482` |
| M3 | `PipelineManager` ghi "Thread-Safe" nhưng **không có lock**; N request đồng thời → N lần nạp index, 1 cái bị vứt | `app.py:94-128` |
| M4 | `ChatOpenAI` **không đặt timeout** → mặc định 600s × 3 retry, treo threadpool | `llm_factory.py:52-62` |
| M5 | `answer_parser.py:30` xoá sạch newline → nhánh render markdown ở `app.js:110-124` là **dead code** | `answer_parser.py:30` |
| M6 | `app.js:183-189` nội suất `innerHTML` thô cho citation card | `static/js/app.js` |
| M7 | Grounding là **lời khuyên mềm**; `FocusedAnswerParser` không phát hiện câu từ chối. Với sản phẩm tra cứu luật, "Không có thông tin" **không phân biệt được với câu trả lời thật**, và `app.js:308` đẩy nó vào history làm ngữ cảnh lượt sau | `batch_rag.py:34-40` |
| M8 | BM25 dedup theo toàn bộ `page_content`; 45 nhóm text trùng byte làm mất ứng viên | `hybrid_retriever.py:50-57` |
| M9 | `app.py:289` truyền `QueryRouter(llm=llm)` — bị `**kwargs` nuốt và vứt đi | `app.py:289` |
| M10 | `build_index.py:129-130` in `health['count']`/`['sources']` nhưng **không đọc `health['healthy']`** — bỏ rơi đúng biến có thể bắt được F1 | `build_index.py:129-130` |

---

## 4. ✅ NHỮNG CHỖ LÀM ĐÚNG — ĐỪNG "DỌN" MẤT

- `app.py:41-45` — CORS **không** dùng `*`. Đúng mặc định an toàn.
- `app.js:88-91` — escape XSS trên câu trả lời, đúng thứ tự thao tác.
- `app.py:235-239` — endpoint `def` đồng bộ kèm comment giải thích: tránh blocking event loop.
- Embedding singleton thật — bge-m3 không nạp lại mỗi request.
- Công thức RRF chuẩn; nhánh fallback `hybrid_retriever.py:94-103` logic đúng, stateless.
- `reranker.py:62-65` — bounds-check về đúng list gốc, không theo thứ tự response.
- Parse JSON của router **thực sự exception-safe**: sai chữ hoa/thiếu field đều rơi về `direct`.
- **`parent_snippet` đã được dùng đúng** ở `batch_rag.py:52-53`.

### 🔎 Đính chính lần review trước

Lần review 29/09 nói `parent_snippet` *"không nơi nào dùng"*. **Kết luận đó sai:**
`hierarchical_chunker.py:75` sinh ra và `batch_rag.py:52-53` tiêu thụ thật.
Fix đó đã được implement rồi — **đừng xoá `parent_snippet`.**

Chỉ còn `parent_store` (`hierarchical_chunker.py:87`) thật sự bị bỏ rơi:
`build_index.py:85` bind, `:137` chỉ in `len()`, xong mất.

---

## 5. THỨ TỰ SỬA

| # | Việc | Sửa | Thời gian |
| :--: | :--- | :--- | ---: |
| 1 | Index chỉ 12,6% corpus | Bỏ `max_pages=50` hoặc nâng lên 500, in cảnh báo khi cắt | 10 phút |
| 2 | Metrics đo trên 3–14 mẫu | `EVAL_MAX_WORKERS=4`, thêm cột `n_scored` vào CSV + biểu đồ, raise nếu `n_scored < 80%` | 30 phút |
| 3 | CLI chết `KeyError` | `main.py:103,111` thêm `history_block=""` | 5 phút |
| 4 | `/api/benchmark` 404 | Đổi tên file khớp `eval_50_unified_pipeline.csv` | 2 phút |
| 5 | Gemini key lọt log | Chuyển sang header `x-goog-api-key`, bỏ `print(resp.text)` | 15 phút |
| 6 | Reranker/router chết im lặng | Thêm nhánh `else` in status code + cờ `degraded` vào CSV | 30 phút |
| 7 | BM25 nhiễu score 0 | Lọc `bm25_scores > 0` trước `argsort` | 5 phút |
| 8 | `git init` + commit đầu tiên | Không có việc này thì mọi fix trên dễ mất | 5 phút |

---

## 6. ẢNH HƯỞNG TỚI CV

Con số hiện tại (0.6583 / 0.8412 / 0.6292 / 0.6369) **chưa nên ghi lên CV** —
đó là trung bình của 3–14 câu, không phải 50 câu.

Sau khi sửa mục **#1** rồi **#2** và chạy lại, sẽ có số thật đáng ghi.
Trong lúc đó, nên mô tả **kiến trúc + công nghệ** và **không kèm số liệu đo**.

## 7. CẦN CẬP NHẬT TÀI LIỆU

- `ARCHITECTURE.md:90` — xoá dòng `chunking.py` khỏi cây thư mục (file không tồn tại).
- `ARCHITECTURE.md` §5 — đổi mô tả "4 chỉ số Ragas (IR-based)" vì giờ đã là Ragas LLM-judge thật.
- `ARCHITECTURE.md` — fingerprint là SHA-256 **cắt 64 bit**, không phải SHA-256 đầy đủ.
- `README.md:9` — claim QueryRouter cần được đi kèm trạng thái degraded.
