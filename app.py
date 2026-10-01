"""
FastAPI Server & Web Application cho hệ thống Insight into RAG.
Cung cấp REST API hiệu năng cao và giao diện Web chuyên nghiệp.
Kiến trúc: Adaptive Two-Stage Hybrid RAG (Single Unified Pipeline).
  QueryRouter (Gemini) → ChromaDB Dense + BM25 → RRF → Cross-Encoder Reranker → LLM
"""

import os
import sys
import time
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Thêm thư mục gốc vào PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.settings import settings
from src.models.llm_factory import get_llm
from src.models.embedding_factory import get_embeddings
from src.storage.vector_store import HybridVectorDB
from src.retrieval.reranker import CrossEncoderReranker
from src.pipeline.batch_rag import BatchRAG

app = FastAPI(
    title="Insight into RAG API",
    description="Hệ thống tra cứu tài liệu thông minh với Adaptive Two-Stage Hybrid RAG",
    version="2.2.0",
)

# CORS Middleware - Giới hạn origin thay vì allow_origins=["*"]
ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# RATE LIMITING (in-memory, per IP)
# ============================================================
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_MAX = settings.RATE_LIMIT_MAX   # max requests
RATE_LIMIT_WINDOW = settings.RATE_LIMIT_WINDOW  # per N seconds

def check_rate_limit(ip: str):
    now = time.time()
    history = _rate_limit_store[ip]
    # Xóa các request cũ hơn cửa sổ thời gian
    _rate_limit_store[ip] = [t for t in history if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Bạn đã gửi quá {RATE_LIMIT_MAX} yêu cầu trong {RATE_LIMIT_WINDOW} giây. Vui lòng chờ."
        )
    _rate_limit_store[ip].append(now)


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """Làm sạch đầu vào: cắt độ dài, loại bỏ ký tự điều khiển và HTML injection"""
    if not text:
        return ""
    # Giới hạn độ dài
    text = text[:max_length]
    # Loại bỏ ký tự điều khiển ASCII (trừ newline và tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Loại bỏ HTML/script injection cơ bản
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


# ============================================================
# Quản Lý Pipeline Singleton Trong Bộ Nhớ (Thread-Safe Caching)
# ============================================================
class PipelineManager:
    """Quản lý singleton HybridVectorDB — nạp nhanh từ chỉ mục đĩa, không embed lại."""

    def __init__(self):
        self.hybrid_db: Optional[HybridVectorDB] = None
        self.initialized = False

    def ensure_initialized(self):
        if self.initialized:
            return

        from src.storage.vector_store import check_index_health
        health = check_index_health(settings.PERSIST_DIR, min_expected=100)
        if health["healthy"]:
            print(f"[PipelineManager] Chỉ mục hợp lệ: {health['count']} vectors từ {health['sources']} nguồn.")
        else:
            msg = health.get("error") or f"Chỉ có {health['count']} vectors, chưa đủ."
            print(f"[PipelineManager CẢNH BÁO] {msg}")
            print("  Hãy chạy: python build_index.py --clean")

        self.initialized = True

    # Alias để không break code cũ
    def ensure_docs_loaded(self):
        self.ensure_initialized()

    def get_hybrid_db(self) -> HybridVectorDB:
        self.ensure_initialized()
        if self.hybrid_db is not None:
            return self.hybrid_db

        print("[PipelineManager] Nạp HybridVectorDB từ chỉ mục đĩa...")
        emb_fn = get_embeddings()
        self.hybrid_db = HybridVectorDB(
            documents=None,
            embedding=emb_fn,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_dir=settings.PERSIST_DIR,
        )
        return self.hybrid_db

    def reload(self):
        self.initialized = False
        self.hybrid_db = None
        self.ensure_initialized()


pipeline_mgr = PipelineManager()


# ============================================================
# Schemas
# ============================================================
class ConversationTurn(BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng")
    model: Optional[str] = Field(default=None, description="Tên mô hình LLM trên OpenRouter")
    reranker_model: Optional[str] = Field(default=None, description="Tên mô hình Reranker trên OpenRouter")
    api_key: Optional[str] = Field(default=None, description="OpenRouter API Key tùy chỉnh")
    top_k: Optional[int] = Field(default=None, description="Số lượng context muốn trích xuất")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None,
        description="Lịch sử hội thoại (tối đa 10 lượt gần nhất)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        v = sanitize_input(v, max_length=2000)
        if not v:
            raise ValueError("Câu hỏi không được để trống sau khi làm sạch")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v):
        if v is not None:
            v = v.strip()[:200]  # Giới hạn độ dài API key
        return v


class ContextItem(BaseModel):
    id: int
    text: str
    source: str
    page: Optional[int] = None
    category: Optional[str] = None
    tree_path: Optional[str] = None
    section: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: List[ContextItem]
    model: str
    reranker_model: str
    latency_ms: int
    total_contexts: int


# ============================================================
# Endpoints
# ============================================================
@app.get("/api/config")
def get_config():
    """Lấy cấu hình hệ thống và câu hỏi mẫu"""
    sample_questions = []
    testset_path = Path(settings.TESTSET_PATH)
    if testset_path.exists():
        try:
            with open(testset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sample_questions = [item.get("user_input", "") for item in data[:6] if item.get("user_input")]
        except Exception:
            pass

    if not sample_questions:
        sample_questions = [
            "RAG là gì và nó khắc phục những nhược điểm nào của LLM?",
            "Sự khác biệt cốt lõi giữa Recursive Chunking và Semantic Chunking?",
            "Tại sao BM25 kết hợp với Vector Search (Hybrid RRF) lại tối ưu hơn?",
            "Cơ chế hoạt động của HyDE trong việc cải thiện truy vấn là gì?",
            "Mô hình Cross-Encoder Reranker đóng vai trò gì sau khi truy xuất?"
        ]

    doc_files = list(Path(settings.DATA_DIR).glob("**/*.pdf")) + list(Path(settings.DATA_DIR).glob("**/*.docx"))
    unique_docs = sorted(list({f.name for f in doc_files}))


    has_env_key = bool(settings.OPENROUTER_API_KEY and "your" not in settings.OPENROUTER_API_KEY)

    return {
        "status": "ready",
        "has_env_key": has_env_key,
        "default_model": settings.OPENROUTER_MODEL,
        "sample_questions": sample_questions,
        "doc_count": len(unique_docs),
        "doc_files": unique_docs[:10],
    }



# ============================================================
# Chú ý: Dùng def (không dùng async def) để FastAPI chạy
# trong ThreadPool, tránh chặn Event Loop làm treo reload/new tab!
# ============================================================
@app.post("/api/chat", response_model=QueryResponse)
def chat_endpoint(req: QueryRequest, request: Request):
    """
    Xử lý tra cứu câu hỏi qua Adaptive Two-Stage Hybrid RAG pipeline.
    Chạy trong ThreadPool riêng biệt để không chặn máy chủ khi người dùng reload trang!
    Pipeline: QueryRouter → ChromaDB + BM25 → RRF → Cross-Encoder Reranker → LLM
    Tính năng mới: Multi-turn conversation history (nhớ ngữ cảnh hội thoại), Rate Limiting, Input Sanitization.
    """
    # Rate limiting theo IP
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    pipeline_mgr.ensure_docs_loaded()

    start_time = time.time()

    # Xác định API Key và model linh hoạt từ Request (hoặc fallback về .env)
    actual_key = (req.api_key.strip() if req.api_key else "") or settings.OPENROUTER_API_KEY
    if not actual_key or "your" in actual_key:
        raise HTTPException(
            status_code=400,
            detail="Chưa cấu hình OPENROUTER_API_KEY. Vui lòng nhập API Key vào ô cấu hình trên giao diện hoặc file .env!"
        )

    actual_llm_model = (req.model.strip() if req.model else "") or settings.OPENROUTER_MODEL
    actual_rerank_model = (req.reranker_model.strip() if req.reranker_model else "") or settings.RERANKER_MODEL

    # Xây dựng ngữ cảnh hội thoại để đưa vào prompt (tối đa 6 lượt gần nhất)
    history_turns = req.conversation_history or []
    history_turns = history_turns[-6:]  # Giới hạn 6 lượt gần nhất để tránh vượt context window
    history_text = ""
    if history_turns:
        lines = []
        for turn in history_turns:
            role_label = "Người dùng" if turn.role == "user" else "Trợ lý"
            lines.append(f"{role_label}: {sanitize_input(turn.content, max_length=500)}")
        history_text = "\n".join(lines)

    llm = get_llm(model_name=actual_llm_model, api_key=actual_key)

    raw_contexts = []
    answer = ""

    try:
        # ============================================================
        # GIAI ĐOẠN 1: TIỀN XỬ LÝ CÂU HỎI THÍCH ỨNG (QUERY ROUTER)
        # Gemini tự động phân loại: rewrite / decompose / hyde / direct
        # ============================================================
        from src.query_transform.query_router import QueryRouter
        from src.retrieval.hybrid_retriever import TwoStageHybridRetriever

        router = QueryRouter(llm=llm)
        route_result = router.process(req.query)
        strategy = route_result["strategy_used"]
        print(f"[SmartRAG] Phân loại ý định: {strategy.upper()} - {route_result['explanation']}")

        # Khởi tạo kho dữ liệu Hybrid (embedding cố định: BAAI/bge-m3 local)
        hdb = pipeline_mgr.get_hybrid_db()

        # Khởi tạo Cross-Encoder Re-ranker (OpenRouter + Gemini fallback)
        reranker = CrossEncoderReranker(
            model_name=actual_rerank_model,
            top_k=req.top_k or settings.RERANK_TOP_K,
            api_key=actual_key,
        )

        # ============================================================
        # GIAI ĐOẠN 2 & 3: TRUY VẤN VÀ HỢP NHẤT TUẦN TỰ (TWO-STAGE)
        # Bước 1: ChromaDB + BM25 -> RRF (lọc thô 15-20 đoạn trong 0.001s)
        # Bước 2: Cross-Encoder Re-ranker (lọc tinh ra 3-5 đoạn đắt giá nhất)
        # ============================================================
        two_stage_retriever = TwoStageHybridRetriever(
            vector_db=hdb.vector_db,
            bm25=hdb.bm25,
            documents=hdb.documents,
            reranker=reranker,
            candidate_k=settings.CANDIDATE_K,  # Lọc thô lấy đoạn ứng viên chuẩn hóa theo settings
            k=req.top_k or settings.RERANK_TOP_K,  # Lọc tinh đoạn chất lượng nhất cho LLM
        )

        rag = BatchRAG(llm=llm)

        # ============================================================
        # Xử lý theo chiến lược QueryRouter
        # ============================================================
        if strategy == "decompose":
            # Trường hợp câu hỏi phức/so sánh: truy xuất song song cho từng câu con
            sub_queries = route_result.get("processed_queries", [req.query])
            all_candidate_docs = []
            seen_c = set()
            for sq in sub_queries:
                docs = two_stage_retriever.invoke(sq)
                for d in docs:
                    if d.page_content not in seen_c:
                        all_candidate_docs.append(d)
                        seen_c.add(d.page_content)
            # Tái xếp hạng lần cuối cho toàn bộ câu hỏi gốc
            if reranker:
                final_docs = reranker.rerank(req.query, all_candidate_docs)[: (req.top_k or 5)]
            else:
                final_docs = all_candidate_docs[: (req.top_k or 5)]

            formatted_ctx = rag._format_docs(final_docs)
            history_block = f"[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n{history_text}\n\n" if history_text else ""
            prompt_text = rag.prompt.format(history_block=history_block, context=formatted_ctx, question=req.query)
            answer_list = rag.batch_generate([prompt_text])
            results = [{
                "answer": answer_list[0] if answer_list else "Không có thông tin",
                "contexts": [d.page_content for d in final_docs],
                "documents": final_docs
            }]

        elif strategy == "hyde" and route_result.get("hypothetical_doc"):
            # Dùng câu trả lời giả định để tìm kiếm ngữ nghĩa, kết hợp BM25 của câu hỏi gốc
            hypo_text = route_result["hypothetical_doc"]
            docs = two_stage_retriever.invoke(f"{req.query}\n{hypo_text}")
            formatted_ctx = rag._format_docs(docs)
            history_block = f"[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n{history_text}\n\n" if history_text else ""
            prompt_text = rag.prompt.format(history_block=history_block, context=formatted_ctx, question=req.query)
            answer_list = rag.batch_generate([prompt_text])
            results = [{
                "answer": answer_list[0] if answer_list else "Không có thông tin",
                "contexts": [d.page_content for d in docs],
                "documents": docs
            }]

        elif strategy == "rewrite":
            # Chuẩn hóa câu hỏi (viết tắt → thuật ngữ đầy đủ) trước khi tra cứu
            target_q = route_result["processed_queries"][0]
            results = rag.answer_with_contexts_batch([target_q], two_stage_retriever, history_text=history_text)
            req.query = f"{req.query} (Đã tối ưu: {target_q})"

        else:
            # Direct: câu hỏi rõ ràng → tra cứu thẳng qua Two-Stage Retriever
            results = rag.answer_with_contexts_batch([req.query], two_stage_retriever, history_text=history_text)

        raw_docs = []
        if results and len(results) > 0:
            answer = results[0].get("answer", "")
            raw_contexts = results[0].get("contexts", [])
            raw_docs = results[0].get("documents", [])

    except Exception as e:
        print(f"[Error in /api/chat]: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi thực thi RAG pipeline: {str(e)}")

    latency_ms = int((time.time() - start_time) * 1000)

    formatted_contexts: List[ContextItem] = []
    for idx, c in enumerate(raw_contexts, 1):
        doc = raw_docs[idx - 1] if (idx - 1 < len(raw_docs)) else None
        meta = doc.metadata if (doc and hasattr(doc, "metadata")) else {}
        formatted_contexts.append(
            ContextItem(
                id=idx,
                text=c.strip(),
                source=meta.get("source") or "Tài liệu tham khảo",
                page=meta.get("page"),
                category=meta.get("category"),
                tree_path=meta.get("tree_path"),
                section=meta.get("section"),
            )
        )


    return QueryResponse(
        query=req.query,
        answer=answer,
        contexts=formatted_contexts,
        model=actual_llm_model,
        reranker_model=actual_rerank_model,
        latency_ms=latency_ms,
        total_contexts=len(formatted_contexts),
    )


@app.get("/api/documents")
def list_documents():
    """Danh sách các file PDF hiện có trong hệ thống"""
    p = Path(settings.DATA_DIR)
    files = list(p.glob("*.pdf")) + list(p.glob("**/*.pdf"))
    result = []
    for f in sorted(list(set(files))):
        size_mb = round(f.stat().st_size / (1024 * 1024), 2)
        result.append({
            "name": f.name,
            "size_mb": size_mb,
            "path": str(f.relative_to(ROOT_DIR)) if ROOT_DIR in f.parents else f.name
        })
    return {"total": len(result), "documents": result}


@app.get("/api/benchmark")
def get_benchmark_report():
    """Lấy dữ liệu kết quả đánh giá Ragas 50 câu hỏi (benchmark_summary.csv)"""
    csv_path = ROOT_DIR / "reports" / "benchmark_summary.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu benchmark. Hãy chạy run_evaluation_50.py trước.")

    import pandas as pd
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    chart_exists = (ROOT_DIR / "reports" / "benchmark_comparison_full.png").exists()

    return {
        "status": "success",
        "chart_url": "/reports/benchmark_comparison_full.png" if chart_exists else None,
        "table": records,
    }


# ============================================================
# Static Files & SPA Routing
# ============================================================
STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/")
def serve_index():
    """Phục vụ giao diện người dùng đơn trang (SPA)"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "message": "Insight into RAG API đang hoạt động! File static/index.html đang được tạo.",
        "docs_url": "/docs"
    })


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print(" 🚀 INSIGHT INTO RAG - FASTAPI WEB SERVER")
    print(" Giao diện web:   http://localhost:8000")
    print(" API Swagger Docs: http://localhost:8000/docs")
    print("=" * 70 + "\n")
    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
