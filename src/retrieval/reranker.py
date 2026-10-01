"""
Re-ranking trong hệ thống RAG:
Hỗ trợ 2 phương thức:
1. OpenRouter Cross-Encoder Rerank API (mặc định: 'nvidia/llama-nemotron-rerank-vl-1b-v2:free', 'cohere/rerank-v3.5').
2. Native Gemini Reranker qua Google AI Studio API: Dùng chính model Gemini chấm điểm tương quan sâu nếu không có key OpenRouter.
"""

from typing import List, Optional
import requests
import json
from langchain_core.documents import Document
from configs.settings import settings


class CrossEncoderReranker:
    """
    Tái xếp hạng danh sách tài liệu ứng viên để lọc tinh từ 15-20 đoạn xuống 3-5 đoạn đắt giá nhất.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: int = settings.RERANK_TOP_K,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.top_k = top_k
        self.model_name = model_name or settings.RERANKER_MODEL
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        raw_base = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        if raw_base.endswith("/rerank"):
            self.rerank_url = raw_base
        else:
            self.rerank_url = f"{raw_base}/rerank"

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
            response = requests.post(self.rerank_url, headers=headers, json=payload, timeout=25)
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
                # Log rõ ràng thay vì âm thầm trả None
                rate_limit = response.headers.get("X-RateLimit-Remaining", "?")
                print(f"[Reranker] ⚠️  OpenRouter Rerank HTTP {response.status_code} "
                      f"(Rate Limit Remaining: {rate_limit}) — Fallback về RRF thô.")
        except Exception as e:
            print(f"[Reranker] Lỗi gọi OpenRouter Rerank API: {e}")

        return None

    def _rerank_via_gemini(self, query: str, documents: List[Document]) -> Optional[List[Document]]:
        """Tái xếp hạng bằng Google AI Studio Gemini API (LLM-as-a-Reranker)"""
        gemini_key = getattr(settings, "GEMINI_API_KEY", "")
        if not gemini_key:
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
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
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
                if parts:
                    parsed = json.loads(parts[0].get("text", "{}"))
                    ranked_indices = parsed.get("ranked_indices", [])
                    reranked = [documents[i] for i in ranked_indices if 0 <= i < len(documents)]
                    if reranked:
                        return reranked[: self.top_k]
        except Exception as e:
            print(f"[Reranker] Lỗi gọi Gemini Rerank API: {e}")

        return None

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Thực hiện tái xếp hạng:
        Ưu tiên OpenRouter Reranker API -> Fallback Gemini Reranker -> Giữ nguyên Top RRF.
        """
        if not documents:
            return []

        # 1. Thử gọi OpenRouter Rerank
        results = self._rerank_via_openrouter(query, documents)
        if results:
            return results

        # 2. Thử gọi Gemini Rerank nếu OpenRouter không có key
        results_gemini = self._rerank_via_gemini(query, documents)
        if results_gemini:
            return results_gemini

        # 3. Fallback: Lấy Top k tài liệu từ danh sách RRF ban đầu
        return documents[: self.top_k]


# Alias tương thích ngược
OpenRouterReranker = CrossEncoderReranker
