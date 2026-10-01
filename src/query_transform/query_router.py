"""
LLM Query Router & Advanced Query Rewriter - Thuần Google AI Studio REST API
Sử dụng trực tiếp REST API nguyên bản của Google AI Studio (Gemini):
- Không phụ thuộc vào thư viện OpenAI.
- Phân loại chiến lược bằng Structured JSON Output của Google.
- Viết lại câu chuẩn xác, phân rã câu hỏi (Decompose), sinh văn bản giả định (HyDE).
- Cơ chế tinh giản: Nếu không gọi được API thì truyền thẳng câu hỏi gốc của người dùng, không thêm thắt.
- Toàn bộ tham số timeout, model, endpoint đều đọc tập trung từ settings.
"""

import json
from typing import Literal, Dict, Any, List, Optional
import requests
from pydantic import BaseModel, Field
from configs.settings import settings


class RouteDecision(BaseModel):
    strategy: Literal["decompose", "hyde", "rewrite", "direct"] = Field(
        ...,
        description="""
        Lựa chọn chiến lược xử lý câu hỏi:
        - 'decompose': Dùng cho câu hỏi so sánh, đối chiếu, có nhiều vế hoặc hỏi về 2-3 chủ thể khác nhau.
        - 'hyde': Dùng cho câu hỏi mở, hỏi về nguyên lý, cơ chế hoạt động, định nghĩa trừu tượng cần giải thích sâu.
        - 'rewrite': Dùng cho câu hỏi quá ngắn, viết tắt (vd: bhxh, hđlđ, đđ), dùng từ lóng đời thường (sổ đỏ, đuổi việc, quỵt lương).
        - 'direct': Dùng khi câu hỏi đã rõ ràng, đầy đủ ngữ cảnh, dùng thuật ngữ chuẩn xác, không cần sửa đổi.
        """
    )
    reasoning: str = Field(..., description="Giải thích ngắn gọn 1 câu lý do chọn.")


class LLMQueryRouter:
    """
    Bộ định tuyến và viết lại câu truy vấn bằng Native Google AI Studio API (Gemini).
    Không dùng OpenAI SDK, gọi trực tiếp endpoint REST API chính thức của Google.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        **_kwargs
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self.timeout = timeout if timeout is not None else settings.ROUTER_TIMEOUT
        self.rest_base = settings.GEMINI_REST_URL.rstrip("/")
        # Dùng URL không chứa key — key đặt trong header x-goog-api-key
        self.endpoint_url = f"{self.rest_base}/models/{self.model_name}:generateContent"

    def _call_openrouter_fallback(self, system_instruction: str, user_prompt: str, json_mode: bool = False, temperature: float = 0.0) -> Optional[str]:
        """Tự động chuyển đổi sang OpenRouter khi Google AI Studio gặp sự cố hoặc vượt giới hạn 429"""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            from src.models.llm_factory import get_llm
            prompt_system = system_instruction
            if json_mode:
                prompt_system += "\nBẮT BUỘC: Chỉ trả về định dạng JSON hợp lệ, không có markdown codeblock, không kèm lời mở đầu."
            
            target_fallback = settings.OPENROUTER_FALLBACK_MODEL
            openrouter_llm = get_llm(model_name=target_fallback, temperature=temperature)
            resp = openrouter_llm.invoke([
                SystemMessage(content=prompt_system),
                HumanMessage(content=user_prompt)
            ])
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            # Làm sạch nếu model bọc trong ```json ... ```
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            print(f"  [QueryRouter Fallback] Đã kích hoạt dự phòng sang OpenRouter ({target_fallback}) thành công.")
            return text
        except Exception as e:
            print(f"  [QueryRouter Fallback] Lỗi OpenRouter: {e}")
            return None

    def _call_gemini_api(self, system_instruction: str, user_prompt: str, json_mode: bool = False, temperature: float = 0.0) -> Optional[str]:
        """Gọi Google AI Studio REST API với cơ chế tự động thử model phụ và Fallback sang OpenRouter"""
        candidate_models = [self.model_name]
        for extra in settings.GEMINI_FALLBACK_MODELS:
            if extra not in candidate_models:
                candidate_models.append(extra)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        for model in candidate_models:
            if not self.api_key:
                break
            endpoint_url = f"{self.rest_base}/models/{model}:generateContent"
            payload: Dict[str, Any] = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": temperature}
            }
            if json_mode:
                payload["generationConfig"]["responseMimeType"] = "application/json"

            try:
                resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                elif resp.status_code == 429:
                    print(f"  [QueryRouter] Model '{model}' chạm giới hạn (HTTP 429). Đang chuyển sang phương án dự phòng...")
                else:
                    safe_body = resp.text.replace(self.api_key, "***KEY***") if self.api_key else resp.text
                    print(f"  [QueryRouter] Model '{model}' HTTP {resp.status_code} — {safe_body[:100]}")
            except Exception as e:
                err_msg = str(e).replace(self.api_key, "***KEY***") if self.api_key else str(e)
                print(f"  [QueryRouter] Model '{model}' lỗi kết nối: {err_msg}")

        # Khi toàn bộ model Google AI Studio lỗi -> Chuyển ngay sang OpenRouter Fallback
        return self._call_openrouter_fallback(system_instruction, user_prompt, json_mode=json_mode, temperature=temperature)

    # ============================================================
    # 1. PHÂN LOẠI CHIẾN LƯỢC (ROUTER)
    # ============================================================
    def decide_strategy(self, query: str) -> RouteDecision:
        """Phân tích câu hỏi và quyết định chiến lược xử lý qua LLM với đa tầng dự phòng"""
        system_prompt = (
            "Bạn là một chuyên gia tối ưu truy vấn tìm kiếm (Query Router) cho hệ thống RAG pháp lý và công nghệ. "
            "Nhiệm vụ của bạn là phân tích câu hỏi người dùng và chọn DUY NHẤT 1 chiến lược phù hợp nhất:\n"
            "- 'decompose': Dùng cho câu hỏi so sánh, đối chiếu, có nhiều vế hoặc hỏi về 2-3 chủ thể khác nhau.\n"
            "- 'hyde': Dùng cho câu hỏi mở, hỏi về nguyên lý, cơ chế hoạt động, định nghĩa trừu tượng cần giải thích sâu.\n"
            "- 'rewrite': Dùng cho câu hỏi quá ngắn, viết tắt (vd: bhxh, hđlđ, đđ), dùng từ lóng đời thường (sổ đỏ, đuổi việc, quỵt lương).\n"
            "- 'direct': Dùng khi câu hỏi đã rõ ràng, đầy đủ ngữ cảnh, dùng thuật ngữ chuẩn xác, không cần sửa đổi.\n\n"
            "BẮT BUỘC TRẢ VỀ JSON VỚI 2 TRƯỜNG: 'strategy' và 'reasoning'."
        )

        user_prompt = f"Phân tích câu hỏi sau:\n\"\"\"{query}\"\"\""

        raw_json = self._call_gemini_api(system_prompt, user_prompt, json_mode=True, temperature=0.0)

        if raw_json:
            try:
                clean_json = raw_json.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                data = json.loads(clean_json.strip())
                return RouteDecision(**data)
            except Exception as e:
                print(f"[QueryRouter] Lỗi parse JSON: {e} | Raw: {raw_json[:100]}")

        # Chỉ khi tất cả LLM đều không phản hồi thì mới fallback về direct
        return RouteDecision(strategy="direct", reasoning="Toàn bộ kết nối API LLM đều gián đoạn, truyền thẳng câu hỏi gốc.")

    # ============================================================
    # 2. PHƯƠNG PHÁP VIẾT LẠI CÂU BẰNG LLM (QUERY REWRITING)
    # ============================================================
    def rewrite_query(self, query: str, reasoning: Optional[str] = None) -> str:
        """
        Viết lại câu hỏi người dùng qua Google AI Studio:
        - Mở rộng từ viết tắt (đđ -> Đất đai, bhxh -> Bảo hiểm xã hội, hđlđ -> Hợp đồng lao động,...).
        - Chuẩn hóa thuật ngữ dân gian/khẩu ngữ sang thuật ngữ pháp lý chính thức.
        - Bổ sung từ khóa mở rộng cho BM25 và Vector Search.
        """
        system_prompt = (
            "Bạn là một chuyên gia tra cứu thông tin (Search Specialist) về Pháp luật Việt Nam và Công nghệ Trí tuệ Nhân tạo.\n"
            "Nhiệm vụ của bạn là VIẾT LẠI câu hỏi của người dùng thành một câu truy vấn tra cứu chuẩn hóa, rõ nghĩa và giàu từ khóa chuyên môn nhất.\n\n"
            "Yêu cầu:\n"
            "1. Giữ nguyên ý định gốc của người dùng.\n"
            "2. Mở rộng toàn bộ các từ viết tắt thông dụng: 'đđ' -> 'Đất đai', 'bhxh' -> 'Bảo hiểm xã hội', 'hđlđ' -> 'Hợp đồng lao động', 'đk' -> 'Điều kiện'.\n"
            "3. Chuyển đổi từ ngữ khẩu ngữ/đời thường sang thuật ngữ pháp lý chính thức:\n"
            "   - 'sổ đỏ / sổ hồng' -> 'Giấy chứng nhận quyền sử dụng đất, quyền sở hữu tài sản gắn liền với đất'.\n"
            "   - 'sa thải / đuổi việc' -> 'đơn phương chấm dứt hợp đồng lao động / kỷ luật sa thải'.\n"
            "   - 'quỵt lương / nợ lương' -> 'chậm trả hoặc không thanh toán tiền lương'.\n"
            "4. Thêm các từ khóa ngữ cảnh cốt lõi liên quan (ví dụ: 'Luật Đất đai', 'Bộ luật Lao động', 'quy định pháp luật hiện hành').\n"
            "5. CHỈ TRẢ VỀ DUY NHẤT 1 CÂU TRUY VẤN ĐÃ TỐI ƯU HÓA, KHÔNG GIẢI THÍCH GÌ THÊM."
        )

        user_prompt = f"Câu hỏi gốc: {query}\n"
        if reasoning:
            user_prompt += f"Lý do phân tích: {reasoning}\n"
        user_prompt += "Câu truy vấn tối ưu:"

        rewritten = self._call_gemini_api(system_prompt, user_prompt, json_mode=False, temperature=0.0)

        # Nếu không gọi được API thì trả thẳng câu hỏi gốc, không thêm thắt
        if rewritten and len(rewritten.strip()) > 3:
            return rewritten.strip().strip('"').strip("'")
        return query

    # ============================================================
    # 3. PHÂN TÁCH CÂU HỎI CON (DECOMPOSITION)
    # ============================================================
    def decompose_query(self, query: str) -> List[str]:
        """Phân tách câu hỏi phức thành các câu hỏi con đơn giản qua Google AI Studio"""
        system_prompt = (
            "Bạn là chuyên gia phân rã truy vấn (Query Decomposition). "
            "Hãy chia câu hỏi phức tạp hoặc câu hỏi so sánh của người dùng thành 2 hoặc 3 câu hỏi con đơn giản, "
            "mỗi câu tập trung vào một khía cạnh độc lập và sử dụng thuật ngữ chuẩn xác.\n"
            "Chỉ trả về danh sách các câu hỏi con, mỗi câu trên 1 dòng, không đánh số, không kèm lời mở đầu."
        )

        result = self._call_gemini_api(system_prompt, f"Câu hỏi: {query}", json_mode=False, temperature=0.0)

        if result:
            lines = [l.strip() for l in result.split("\n") if l.strip()]
            cleaned = []
            for line in lines:
                c = line.lstrip("0123456789.-*•) ").strip()
                if c:
                    cleaned.append(c)
            if cleaned:
                return cleaned[:3]

        # Nếu không gọi được API thì trả về danh sách chỉ chứa câu hỏi gốc
        return [query]

    # ============================================================
    # 4. SINH TÀI LIỆU GIẢ ĐỊNH (HyDE)
    # ============================================================
    def generate_hyde_doc(self, query: str) -> str:
        """Sinh đoạn văn bản giả định mang phong cách tài liệu chuyên ngành qua Google AI Studio"""
        system_prompt = (
            "Hãy viết một đoạn văn ngắn (khoảng 100 - 150 từ) giải thích câu hỏi sau như thể trích từ tài liệu chuyên môn, "
            "giáo trình hoặc văn bản quy phạm pháp luật. Sử dụng từ vựng chuẩn xác và phong cách khẳng định khách quan. "
            "Không mở đầu bằng 'Theo tài liệu' hay các cụm từ thừa."
        )

        result = self._call_gemini_api(system_prompt, f"Câu hỏi: {query}", json_mode=False, temperature=0.1)

        # Nếu không gọi được API thì trả về chính câu hỏi
        if result and len(result.strip()) > 10:
            return result.strip()
        return query

    # ============================================================
    # 5. TIẾN TRÌNH XỬ LÝ TOÀN DIỆN (PROCESS PIPELINE)
    # ============================================================
    def process(self, query: str, forced_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Thực hiện toàn bộ quy trình: Phân loại ý định -> Thực thi biến đổi.
        Nếu gặp sự cố kết nối, tự động truyền thẳng câu hỏi gốc vào hệ thống, không thêm thắt.
        """
        if forced_strategy and forced_strategy != "adaptive":
            decision = RouteDecision(strategy=forced_strategy, reasoning=f"Chỉ định thủ công chiến lược {forced_strategy}")
        else:
            decision = self.decide_strategy(query)

        strategy = decision.strategy
        reasoning = decision.reasoning

        if strategy == "rewrite":
            rewritten_q = self.rewrite_query(query, reasoning)
            return {
                "strategy_used": "rewrite",
                "reasoning": reasoning,
                "original_query": query,
                "processed_queries": [rewritten_q],
                "hypothetical_doc": None,
                "explanation": f"Đã viết lại câu chuẩn xác: '{query}' -> '{rewritten_q}'."
            }

        elif strategy == "decompose":
            sub_queries = self.decompose_query(query)
            all_queries = list(set(sub_queries + [query]))
            return {
                "strategy_used": "decompose",
                "reasoning": reasoning,
                "original_query": query,
                "processed_queries": all_queries,
                "sub_questions": sub_queries,
                "hypothetical_doc": None,
                "explanation": f"Đã phân rã thành {len(sub_queries)} câu hỏi con."
            }

        elif strategy == "hyde":
            hypo_doc = self.generate_hyde_doc(query)
            return {
                "strategy_used": "hyde",
                "reasoning": reasoning,
                "original_query": query,
                "processed_queries": [query],
                "hypothetical_doc": hypo_doc,
                "explanation": "Đã tạo đoạn văn bản giả định (HyDE)."
            }

        else:
            # DIRECT: Truyền thẳng câu hỏi gốc của người hỏi vào luôn
            return {
                "strategy_used": "direct",
                "reasoning": reasoning,
                "original_query": query,
                "processed_queries": [query],
                "hypothetical_doc": None,
                "explanation": "Truyền thẳng câu hỏi gốc của người dùng vào hệ thống tra cứu."
            }


# Alias tương thích
QueryRouter = LLMQueryRouter