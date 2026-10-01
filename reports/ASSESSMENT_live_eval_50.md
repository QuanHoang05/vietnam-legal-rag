# Đánh giá đầu ra đo — `live_eval_50_hybrid_rrf` (bản 2, sau khi dựng lại chỉ mục)

> **Cập nhật:** 27/09/2026 · bản 1 đánh giá lúc 22:33 (chỉ mục 20 vector), bản này đánh giá bản chạy lúc 22:55 (chỉ mục 9.598 vector).
> **Căn cứ:** đọc CSV mới, truy vấn `chroma_data/chroma.sqlite3`, đọc lại mã nguồn sau khi bạn sửa.
> **Không chạy lại gì** — mọi con số dưới đây đo từ dữ liệu sẵn có trên đĩa.

---

## KẾT LUẬN

> **Hệ thống đã được sửa đúng và giờ chạy thật: 7/10.**
> **Nhưng bộ đo trong bản chạy này KHÔNG phải Ragas — đó là một heuristic tự viết dựa trên trùng từ khoá.**
> **Vì vậy: tuyệt đối không ghi "Ragas đạt 0.91 / 0.90 / 0.59" lên CV.**

Tin tốt: lỗi chỉ mục đã được sửa triệt để, câu trả lời thật đã xuất hiện, và các bảo vệ bạn thêm đều có hiệu lực. Phần còn lại là một vấn đề duy nhất nhưng rất nghiêm trọng về uy tín: **tên metric nói Ragas, code chạy đường khác.**

---

## 1. Đã sửa được gì — bằng chứng

### 1.1 Chỉ mục đã lành

| | Trước (22:33) | Sau (22:55) |
| :--- | ---: | ---: |
| Số vector | **20** | **9.598** |
| Số tài liệu nguồn | **1** | **12** |
| `bm25_cache.pkl` | 7,8 KB | 2,2 MB |

Phân bố theo nguồn (truy vấn trực tiếp sqlite):

```
31_2024_QH15_523642.docx (Luật Đất Đai) ............ 3.684
45_2019_QH14_333670.docx (Luật Lao Động) .......... 1.370
LLM.pdf ............................................   990
[Reading]-RAG-System.pdf ...........................   546
10 file PDF còn lại ............................... 3.008
```

### 1.2 Hệ thống bắt đầu trả lời được

| | Trước | Sau | Δ |
| :--- | ---: | ---: | ---: |
| Số câu trả lời có nội dung thật | **0/50** | **38/50 (76%)** | +38 |
| Faithfulness | 0.760 | **0.9067** | +0.147 |
| Answer Relevancy | 0.200 | **0.7803** | +0.580 |
| Context Precision | 0.7055 | **0.9021** | +0.197 |
| Context Recall | 0.2293 | **0.5871** | +0.358 |

Đạt đủ cả 4 ngưỡng tôi đặt ra ở bản 1: ≥70% câu có câu trả lời thật, Answer Relevancy std > 0 (0.279), Context Recall > 0.50.

### 1.3 Các bảo vệ bạn thêm đều hoạt động

Đã kiểm tra trong mã nguồn, tất cả đều có mặt và đúng logic:

| Hạng mục | Vị trí | Trạng thái |
| :--- | :--- | :--- |
| Hàm kiểm tra sức khoẻ chỉ mục (đếm vector + số nguồn) | `vector_store.py:33-56` | Đã thêm |
| Thoát sớm chỉ khi `count >= len(documents)` | `vector_store.py:97-102`, `:193-194` | Đã sửa |
| Cache BM25 + documents kèm fingerprint | `vector_store.py:277-279` | Đã thêm |
| `raise RuntimeError` khi chỉ mục ghi thiếu | `build_index.py:129-133` | Đã thêm |
| Cảnh báo đỏ khi tỷ lệ từ chối > 35% | `ragas_evaluator.py:228-237` | Đã thêm |
| Đường ngưỡng 0.50 trên biểu đồ | `visualizer.py:39` | Đã thêm |

Bản đánh giá này xác nhận các sửa đổi đó hoạt động: lần này chỉ mục lớn hơn 480 lần và hệ thống trả lời được.

---

## 2. VẤN ĐỀ CHẶN — số liệu này không phải do Ragas tính

### 2.1 Bằng chứng

`src/evaluation/ragas_evaluator.py:139`

```python
def evaluate_ragas(..., force_ragas: bool = False) -> ...:
```

`src/evaluation/ragas_evaluator.py:163`

```python
if has_valid_key and force_ragas:     # ← mặc định False
    ...gọi Ragas thật...
```

`src/evaluation/ragas_evaluator.py:225`

```python
result = _compute_real_ir_metrics(eval_subset)   # ← luồng mặc định
```

Còn ở nơi gọi, `run_evaluation_50.py:118` gọi `evaluate_ragas(eval_df)` — **không truyền `force_ragas`**. Vậy toàn bộ số liệu trong CSV và biểu đồ lần này đến từ `_compute_real_ir_metrics()` (dòng 30-131), một hàm tự viết:

| Metric | Cách `_compute_real_ir_metrics` tính |
| :--- | :--- |
| Faithfulness | Tỉ lệ câu trong câu trả lời có ≥35% từ trùng với ngữ cảnh lấy về |
| Answer Relevancy | Công thức tự định nghĩa `0.4 + 0.6 × (tỉ lệ từ câu hỏi xuất hiện trong câu trả lời)`; nếu là câu từ chối thì gán cứng `0.2` (dòng 106-109) |
| Context Precision | Tỉ lệ từ của ngữ cảnh lấy về nằm trong `reference_contexts` |
| Context Recall | Tỉ lệ từ của `reference_contexts` nằm trong ngữ cảnh lấy về |

Hàm còn tự trả về `engine: "live_ir_metrics"` (dòng 130) và log tiêu đề là **"KẾT QUẢ ĐÁNH GIÁ THỰC TẾ (REAL METRICS)"** (dòng 239) — chính tác giả cũng không gọi nó là Ragas.

Một dấu hiệu thống kê nữa: ở lần chạy Ragas thật trước đó, Answer Relevancy có std = **0.000** (đúng điểm sàn của Ragas). Lần này std = 0.279 với trung vị 0.897 — đây là phân phối của công thức tuyến tính, không phải phân phối LLM-judge của Ragas.

### 2.2 Tác động

- Hệ thống **thật sự đã tốt lên** — chỉ số này không phải bịa, nó phản ánh đúng việc truy xuất đã cải thiện. Vấn đề là **thước đo**.
- Nhưng nếu ghi "Ragas: Faithfulness 0.91, Context Recall 0.59" lên CV, người đọc sẽ hiểu là bạn đã chấm bằng LLM-judge. Khi họ hỏi "chạy lại bằng Ragas được không?", câu trả lời hiện tại là **không chạy Ragas được** — đó là điểm trừ lớn về uy tín, đúng loại rủi ro mà bản 1 cảnh báo.
- Trong CV, đây là ranh giới giữa "biết đo" và "đo sai cách rồi gọi là đúng tên".

### 2.3 Cách sửa

**a) Chạy lại bằng Ragas thật** — thêm cờ ở CLI:

```python
# run_evaluation_50.py
parser.add_argument("--force-ragas", action="store_true",
                    help="Bắt buộc chấm bằng Ragas LLM-judge thay vì heuristic trùng từ khoá")
...
eval_result = evaluate_ragas(eval_df, force_ragas=args.force_ragas)
```

```powershell
python run_evaluation_50.py --mode hybrid_rrf --force-ragas
```

**b) Ghi engine vào mọi đầu ra.** Bắt buộc, để không ai nhầm:

```python
detailed_df["metric_engine"] = eval_result.get("engine", "ragas")
```

và đưa vào tiêu đề biểu đồ:

```python
title = f"50-Questions Evaluation ({mode.upper()}) · engine={eval_result.get('engine','ragas')}"
```

**c) Đổi tên hàm cho trung thực.** `evaluate_ragas` hiện vừa là tên Ragas vừa chạy heuristic. Tách rõ:

| Tên mới | Ý nghĩa |
| :--- | :--- |
| `evaluate_with_ragas(df)` | Chỉ Ragas, không có fallback. Lỗi thì raise. |
| `compute_ir_overlap_metrics(df)` | Heuristic trùng từ khoá. Ghi rõ là proxy, không phải Ragas. |
| `evaluate(df, engine="ragas")` | Dispatcher, mặc định `ragas`. |

**d) Nếu buộc phải giữ heuristic**, ghi nó thành metric riêng có tên khác: `KeywordOverlapRecall` thay vì `context_recall`, và kèm câu *"proxy metric, không tương đương Ragas"*. Đừng dùng lại 4 tên chuẩn của Ragas cho thứ không phải Ragas.

---

## 3. Chất lượng còn lại — ngoài vấn đề metric

### 3.1 Phân bố điểm khá tốt, nhưng có hai nhóm yếu rõ

| Nhóm | n | Context Recall | Số câu từ chối | Đánh giá |
| :--- | ---: | ---: | ---: | :--- |
| LuatLaoDong | 10 | 0.710 | 1 | Tốt |
| LuatDatDai | 13 | 0.693 | 1 | Tốt |
| RAG_Theory | 19 | 0.539 | **7** | Yếu nhất về truy xuất |
| AI_Tech | 8 | **0.374** | 3 | Yếu |

Lý do có thể thấy rõ từ chỉ mục: **Luật Đất Đai + Luật Lao Động chiếm 5.054/9.598 chunk (53%)**, còn `[Reading]-RAG-System.pdf` chỉ 546 chunk (5,7%) và `AIO2025_FastAPI_v2.pdf` 258 chunk. Corpus lệch mạnh, các câu hỏi RAG_Theory và AI_Tech bị "chìm" trong đám văn bản luật.

### 3.2 Context Recall 0.587 vẫn là nút thắt

13/50 câu dưới 0.5, 36/50 câu dưới 0.7. Nguyên nhân cấu trúc: hệ thống chỉ lấy **k=7 chunk** (`HYBRID_K`) từ **9.598 chunk**. Với tỉ lệ 1:1.371, không thể bao phủ tài liệu đích.

Ba hướng nâng, theo ROI:

1. **Tăng `HYBRID_K` lên 12–15 rồi rerank xuống 3–4** (`RETRIEVE_K`/`RERANK_TOP_K` sẵn có trong `settings.py`). Đây là cách đúng nhất: lấy rộng, lọc sắc.
2. **Dùng parent chunk khi sinh câu trả lời.** Đây là lỗi thiết kế còn sót: `HierarchicalTreeChunker` tạo `parent_store` (1.024 ký tự) và gắn `parent_snippet` vào metadata, nhưng **không nơi nào dùng** — `grep parent_snippet` chỉ ra khỏi chính file chunker, `build_index.py:153` chỉ in ra số lượng. Hiện tại LLM chỉ nhận 350 ký tự của child chunk. Mở rộng sang parent sẽ tăng recall rõ rệt mà không đổi gì khác.
3. **Cân bằng corpus** nếu muốn nhóm RAG_Theory ngang các nhóm khác: giảm `max_pages=50` trong `document_loader.py:23` cho các file luật (3.684 chunk từ 1 file luật là rất nhiều so với 546 chunk cho toàn bộ tài liệu RAG).

### 3.3 Câu từ chối có context đúng vẫn bị từ chối

id=24 (nghỉ phép năm, Luật Lao Động) đạt recall 0.79 nhưng LLM vẫn trả "Không có thông tin". Nghĩa là đoạn lấy về có liên quan nhưng **không chứa đúng điều khoản** → đây đúng là trường hợp child chunk 350 ký tự quá ngắn, củng cố lập luận ở mục 3.2 phương án 2.

### 3.4 Biểu đồ đã khá hơn nhưng còn thiếu

Đã có: đường ngưỡng 0.50, màu phân biệt, nhãn tiếng Việt, 4 chữ số thập phân. Vẫn thiếu:

- **Cột Baseline để so sánh** — bốn cột đứng riêng không cho biết tốt hay xấu. Đây là thiếu sót lớn nhất còn lại của biểu đồ.
- **Ghi `n = 50` và tên engine** lên tiêu đề.
- **Biểu đồ theo nhóm domain** — 4 nhóm có chất lượng rất khác nhau (recall 0.374 → 0.710), con số trung bình đang che mất điều đó.
- **Phân phối từng câu** — trung bình đã giấu 12 câu từ chối.

---

## 4. Đánh giá tổng

| Hạng mục | Bản 1 | Bản 2 | Ghi chú |
| :--- | :---: | :---: | :--- |
| Chuỗi chỉ mục (indexing) | 1/10 | **9/10** | 9.598 vector, 12 nguồn, có fingerprint + guard |
| Truy xuất | 1/10 | **7/10** | Recall 0.587, còn lệch theo domain |
| Chất lượng câu trả lời | 0/10 | **7/10** | 38/50 trả lời thật |
| Bộ đo (metrics) | 2/10 | **3/10** | Đã bỏ hằng số giả, nhưng đang dùng heuristic mang tên Ragas |
| Biểu đồ & báo cáo | 4/10 | **6/10** | Có ngưỡng, thiếu baseline và phân nhóm |
| Tổng | **4/10** | **7/10** | |

**Đánh giá thẳng:** bạn đã sửa đúng trọng tâm và kết quả rất rõ. Nhưng dự án hiện ở trạng thái "hệ thống chạy tốt, bộ đo chưa đúng tên". Sửa mục 2.3 rồi chạy lại `--force-ragas` là con số mới đủ sức để viết lên CV.

---

## 5. Nếu muốn viết lên CV — chỉ khi đã làm mục 2.3

Hiện tại **chưa nên viết số**. Sau khi chạy `--force-ragas` và có số thật:

```
• Xây dựng hệ thống RAG tiếng Việt trên 9.598 chunk từ 12 tài liệu (PDF/DOCX, gồm corpus luật),
  dùng Hybrid Retrieval (BM25 + Dense, hợp nhất RRF) và tự định nghĩa kho chỉ mục ngoại tuyến.
• Đánh giá trên 50 câu hỏi có ground truth theo 4 nhóm domain bằng Ragas
  (Faithfulness / Answer Relevancy / Context Precision / Context Recall).
• Phát hiện và sửa lỗi chỉ mục dở dang làm hệ thống mất 99,8% dữ liệu truy xuất;
  bổ sung cơ chế kiểm tra sức khoẻ chỉ mục và fingerprint cache để lỗi này không tái diễn.
```

Bullet thứ ba là điểm mạnh nhất và nên giữ: nó kể đúng một sự cố thật đã tìm ra và sửa được — thứ mà CV nhiều người không có.

---

## 6. Thứ tự làm tiếp

| # | Việc | Effort | Chặn CV? |
| ---: | :--- | :--- | :--- |
| 1 | Tách `evaluate_with_ragas` / `compute_ir_overlap_metrics`, thêm `--force-ragas` | 30 phút | **Có** |
| 2 | Chạy `python run_evaluation_50.py --mode hybrid_rrf --force-ragas` | ~20 phút | **Có** |
| 3 | Ghi `engine` vào CSV + tiêu đề biểu đồ | 15 phút | Không |
| 4 | Thêm cột Baseline và biểu đồ theo nhóm domain | 1 giờ | Không |
| 5 | Tăng `HYBRID_K` 12–15 + rerank; dùng parent chunk khi sinh câu trả lời | 2–3 giờ | Không |
| 6 | Chạy lại các mode khác (bm25, baseline, decomposition) để có bảng so sánh thật | ~1 giờ | Không |

Bước 6 mới là thứ biến dự án từ "một hệ thống chạy được" thành "một nghiên cứu so sánh phương pháp" — và đó mới là thứ đáng đưa lên CV.
