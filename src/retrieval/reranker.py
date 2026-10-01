"""
Re-ranking trong hệ thống Legal RAG:
Hỗ trợ các phương thức:
1. Local Vietnamese Reranker (Mặc định: 'AITeamVN/Vietnamese_Reranker' chạy cục bộ bằng PyTorch/Transformers, không tốn API, tốc độ cao trên CPU/GPU).
2. OpenRouter Cross-Encoder Rerank API (Dành cho mô hình đám mây: 'nvidia/llama-nemotron-rerank-vl-1b-v2:free', 'cohere/rerank-v3.5').
3. Gemini Reranker qua Google AI Studio API: LLM-as-a-Reranker dự phòng.
Toàn bộ tham số cấu hình đều được đọc tập trung từ settings.
"""

from typing import List, Optional
import sys
import requests
import json
import torch
from langchain_core.documents import Document
from configs.settings import settings

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class LocalVietnameseReranker:
    """
    Bộ tái xếp hạng tiếng Việt chuyên sâu chạy offline trực tiếp trên máy.
    Sử dụng kiến trúc Cross-Encoder Sequence Classification của AITeamVN.
    """
    _instance = None
    _tokenizer = None
    _model = None

    def __init__(self, model_name: Optional[str] = None, max_length: Optional[int] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.max_length = max_length or settings.RERANKER_MAX_LENGTH
        self._ensure_loaded()

    def _ensure_loaded(self):
        if LocalVietnameseReranker._model is None:
            print(f"[LocalReranker] Đang nạp mô hình cục bộ '{self.model_name}'...")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            LocalVietnameseReranker._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            LocalVietnameseReranker._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            LocalVietnameseReranker._model.eval()
            print(f"[LocalReranker] Nạp mô hình '{self.model_name}' thành công!")
        self.tokenizer = LocalVietnameseReranker._tokenizer
        self.model = LocalVietnameseReranker._model

    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        actual_top_k = top_k or settings.RERANK_TOP_K
        if not documents:
            return []
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=self.max_length
                )
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float().tolist()

            if isinstance(scores, float):
                scores = [scores]

            scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:actual_top_k]]
        except Exception as e:
            print(f"[LocalReranker] ⚠️ Lỗi suy luận rerank cục bộ: {e} — Fallback về RRF thô.")
            return documents[:actual_top_k]


class CrossEncoderReranker:
    """
    Tái xếp hạng danh sách tài liệu ứng viên để lọc tinh từ 15-20 đoạn xuống 3-5 đoạn đắt giá nhất.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.top_k = top_k or settings.RERANK_TOP_K
        self.model_name = model_name or settings.RERANKER_MODEL
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.timeout = timeout or settings.RERANKER_TIMEOUT
        raw_base = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        if raw_base.endswith("/rerank"):
            self.rerank_url = raw_base
        else:
            self.rerank_url = f"{raw_base}/rerank"

        self.local_reranker = None
        if "Vietnamese_Reranker" in self.model_name or "AITeamVN" in self.model_name:
            try:
                self.local_reranker = LocalVietnameseReranker(self.model_name)
            except Exception as e:
                print(f"[Reranker] Không thể khởi tạo Local Reranker: {e}")

    def _rerank_via_openrouter(self, query: str, documents: List[Document]) -> Optional[List[Document]]:
        """Tái xếp hạng qua OpenRouter Rerank API"""
        if not self.api_key or "your" in self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.OPENROUTER_REFERER,
            "X-Title": settings.OPENROUTER_TITLE,
        }

        doc_texts = [doc.page_content for doc in documents]
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": self.top_k,
        }

        try:
            response = requests.post(self.rerank_url, headers=headers, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                reranked_docs = []
                for item in results:
                    idx = item.get("index")
                    if idx is not None and 0 <= idx < len(documents):
                        reranked_docs.append(documents[idx])

                if reranked_docs:
                    return reranked_docs[: self.top_k]
            else:
                rate_limit = response.headers.get("X-RateLimit-Remaining", "?")
                print(f"[Reranker] ⚠️  OpenRouter Rerank HTTP {response.status_code} "
                      f"(Rate Limit Remaining: {rate_limit}) — Chuyển sang dự phòng.")
        except Exception as e:
            print(f"[Reranker] Lỗi gọi OpenRouter Rerank API: {e}")

        return None

    def _rerank_via_gemini(self, query: str, documents: List[Document]) -> Optional[List[Document]]:
        """Tái xếp hạng bằng Google AI Studio Gemini API (LLM-as-a-Reranker)"""
        gemini_key = settings.GEMINI_API_KEY
        if not gemini_key:
            return None

        candidate_models = [settings.GEMINI_MODEL]
        for fb in settings.GEMINI_FALLBACK_MODELS:
            if fb not in candidate_models:
                candidate_models.append(fb)

        rest_base = settings.GEMINI_REST_URL.rstrip("/")
        for model in candidate_models:
            url = f"{rest_base}/models/{model}:generateContent?key={gemini_key}"
            doc_snippets = []
            for idx, doc in enumerate(documents):
                clean_text = doc.page_content.strip().replace("\n", " ")[:250]
                doc_snippets.append(f"[{idx}]: {clean_text}")

            system_prompt = (
                "Bạn là bộ tái xếp hạng văn bản chuyên biệt (Cross-Encoder Re-ranker). "
                "Nhiệm vụ: Chấm điểm và sắp xếp thứ tự các đoạn trích từ liên quan nhiều nhất đến ít nhất đối với câu hỏi.\n"
                "Chỉ trả về JSON có trường 'ranked_indices' chứa danh sách các số index (tối đa 5 index), ví dụ: {\"ranked_indices\": [2, 0, 4]}."
            )

            user_prompt = f"Câu hỏi: {query}\n\nDanh sách đoạn trích:\n" + "\n".join(doc_snippets)
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json"
                }
            }

            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
                    if parts:
                        parsed = json.loads(parts[0].get("text", "{}"))
                        ranked_indices = parsed.get("ranked_indices", [])
                        reranked = [documents[i] for i in ranked_indices if 0 <= i < len(documents)]
                        if reranked:
                            return reranked[: self.top_k]
            except Exception:
                pass

        return None

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Thực hiện tái xếp hạng theo thứ tự ưu tiên:
        1. Local Vietnamese Reranker (nếu cấu hình hoặc có sẵn)
        2. OpenRouter Reranker API
        3. Gemini LLM Reranker
        4. Fallback giữ nguyên Top RRF ban đầu
        """
        if not documents:
            return []

        # 1. Ưu tiên Local Reranker
        if self.local_reranker:
            return self.local_reranker.rerank(query, documents, self.top_k)

        # 2. Thử gọi OpenRouter Rerank
        results = self._rerank_via_openrouter(query, documents)
        if results:
            return results

        # 3. Thử gọi Gemini Rerank
        results_gemini = self._rerank_via_gemini(query, documents)
        if results_gemini:
            return results_gemini

        # 4. Fallback: Lấy Top k tài liệu từ danh sách RRF ban đầu
        return documents[: self.top_k]
