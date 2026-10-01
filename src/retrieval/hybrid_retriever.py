"""
Chiến lược Hybrid Search & Two-Stage Retrieval:
Quy trình phối hợp tuần tự chuẩn mực theo yêu cầu kiến trúc:
- Giai đoạn 1: Tiền xử lý câu hỏi qua QueryRouter (Decompose, HyDE, Rewrite, hoặc Direct).
- Giai đoạn 3: Chạy cả hai bước nối tiếp nhau:
    Bước 1 — RRF (Reciprocal Rank Fusion):
        ChromaDB (Dense) + BM25 (Sparse) -> RRF gộp kết quả thô lấy Top 15-20 ứng viên trong 0.001s.
    Bước 2 — Cross-Encoder Re-ranker:
        Chấm điểm tương quan ngữ nghĩa sâu từng cặp [Câu hỏi + Đoạn trích], tinh lọc từ 15-20 đoạn xuống 3-5 đoạn đắt giá nhất cho LLM đọc.
"""

import numpy as np
from collections import defaultdict
from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, ConfigDict
from configs.settings import settings




class HybridRetriever(BaseRetriever):
    """
    Two-Stage Hybrid Search chuẩn mực:
    Bước 1 — RRF (Reciprocal Rank Fusion):
        Gộp Dense Vector (ChromaDB) và Sparse Keyword (BM25) để lọc thô 15-20 đoạn ứng viên trong 0.001 giây.
    Bước 2 — Cross-Encoder Re-ranker:
        Chấm điểm tương quan ngữ nghĩa sâu từng cặp [Câu hỏi + Đoạn trích], lọc tinh từ 15-20 đoạn xuống 3-5 đoạn đắt giá nhất cho LLM.
    """
    vector_db: Any = Field(description="Chroma Vector DB instance")
    bm25: Any = Field(description="Chỉ mục BM25Okapi")
    documents: List[Any] = Field(description="Danh sách tài liệu tương ứng BM25")
    reranker: Optional[Any] = Field(default=None, description="CrossEncoderReranker instance (Lọc tinh)")
    candidate_k: int = Field(default=settings.CANDIDATE_K, description="Số lượng ứng viên lọc thô qua RRF (mặc định cấu hình qua settings)")
    k: int = Field(default=settings.RERANK_TOP_K, description="Số lượng tài liệu chắt lọc tinh cuối cùng cho LLM (mặc định qua settings)")
    rrf_k: int = Field(default=settings.RRF_K, description="Hằng số làm mượt mẫu số RRF (mặc định 40)")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _reciprocal_rank_fusion(self, results_list: List[List[Document]], k: int) -> List[Document]:
        """
        Tính điểm RRF cho từng tài liệu dựa trên thứ hạng nghịch đảo: sum(1 / (k + rank)).
        Chạy bằng toán học thuần túy trong 0.001 giây, không tốn tài nguyên.
        """
        rrf_scores = defaultdict(float)
        doc_content_map = {}

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.page_content
                rrf_scores[doc_id] += 1.0 / (k + rank)
                doc_content_map[doc_id] = doc

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_content_map[doc_id] for doc_id, score in sorted_docs]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        _run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        results_list = []

        # ============================================================
        # BƯỚC 1: LỌC THÔ BẰNG RRF (CANDIDATE GENERATION)
        # ============================================================
        # 1. Tìm kiếm bằng Vector Dense (Lấy 15-20 đoạn)
        vector_results = self.vector_db.similarity_search(query, k=self.candidate_k)
        results_list.append(vector_results)

        # 2. Tìm kiếm bằng BM25 Sparse (Lấy 15-20 đoạn)
        if self.bm25 is not None and len(self.documents) > 0:
            try:
                from underthesea import word_tokenize
                tokenize_fn = lambda text: word_tokenize(text.lower())
            except ImportError:
                tokenize_fn = lambda text: text.lower().split()

            tokenized_query = tokenize_fn(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Lọc chỉ lấy candidate có score > 0 (tránh nhiễu toàn 0 với query viết tắt/lạ)
            positive_indices = np.where(bm25_scores > 0)[0]
            if len(positive_indices) > 0:
                top_pos_indices = positive_indices[np.argsort(bm25_scores[positive_indices])[::-1]][: self.candidate_k]
                bm25_results = [self.documents[i] for i in top_pos_indices]
            else:
                bm25_results = []  # Không có candidate thật → không thêm nhiễu vào pool
            if bm25_results:
                results_list.append(bm25_results)

        # 3. Gộp bảng xếp hạng bằng RRF thành Top 15-20 ứng viên
        fused_candidates = self._reciprocal_rank_fusion(results_list, k=self.rrf_k)[: self.candidate_k]

        # ============================================================
        # BƯỚC 2: LỌC TINH BẰNG CROSS-ENCODER RE-RANKER
        # ============================================================
        if self.reranker is not None and hasattr(self.reranker, "rerank"):
            try:
                reranked_docs = self.reranker.rerank(query, fused_candidates)
                if reranked_docs:
                    return reranked_docs[: self.k]
            except Exception as e:
                print(f"[HybridRetriever] Lỗi khi gọi Re-ranker: {e}. Fallback dùng Top RRF.")

        # Fallback lấy Top k của RRF nếu không cấu hình Reranker
        return fused_candidates[: self.k]


# Alias đảm bảo tương thích 100%
TwoStageHybridRetriever = HybridRetriever
