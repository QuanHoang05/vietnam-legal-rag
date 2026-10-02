"""
MÔ-ĐUN ĐÁNH GIÁ SINGLE-PASS LLM-AS-A-JUDGE VỚI STRUCTURED OUTPUT (JSON SCHEMA)
Đánh giá đồng thời 4 tiêu chuẩn Ragas (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
trong 1 lần gọi LLM duy nhất (Single-pass), giảm 75% số lượng request, tránh hoàn toàn Rate Limit.
"""

import json
import re
from typing import Dict, Any, List
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.llm_factory import get_evaluator_llm


JUDGE_SYSTEM_PROMPT = """Bạn là một Chuyên gia Thẩm định Pháp lý & Đánh giá Hệ thống RAG (Retrieval-Augmented Generation) độc lập, khách quan và nghiêm ngặt.
Nhiệm vụ của bạn là đánh giá chất lượng câu trả lời của mô hình RAG đối chiếu với tài liệu pháp luật được truy xuất (retrieved_contexts) và đáp án chuẩn (reference).

Bạn phải chấm điểm 4 tiêu chí cốt lõi trên thang điểm từ 0.00 đến 1.00 (lấy 2 chữ số thập phân) kèm theo giải trình khách quan:

1. FAITHFULNESS (Độ trung thực - Thang điểm 0.00 đến 1.00):
   - Mọi thông tin, khẳng định, số liệu trong [CÂU TRẢ LỜI CỦA RAG] có được suy diễn trực tiếp từ [TÀI LIỆU TRUY XUẤT] hay không?
   - 1.00: 100% nội dung trả lời đều có căn cứ trực tiếp trong tài liệu trích dẫn.
   - Trừ điểm: Có yếu tố bịa đặt, suy diễn vượt quá ngữ cảnh hoặc trích dẫn sai số Điều/Khoản luật.

2. ANSWER_RELEVANCY (Độ phù hợp với câu hỏi - Thang điểm 0.00 đến 1.00):
   - [CÂU TRẢ LỜI CỦA RAG] có trả lời đúng, trúng và giải quyết trọn vẹn câu hỏi trong [CÂU HỎI NGƯỜI DÙNG] hay không?
   - 1.00: Trả lời đi thẳng vào vấn đề, mạch lạc, giải quyết triệt để yêu cầu của người dùng.
   - Trừ điểm: Trả lời vòng vo, lạc đề, hoặc trả lời câu "Không có thông tin" khi tài liệu có dữ kiện.

3. CONTEXT_PRECISION (Độ chính xác xếp hạng tài liệu - Thang điểm 0.00 đến 1.00):
   - Đo lường xem các đoạn văn bản luật thực sự liên quan nhất có xuất hiện ở các vị trí đầu tiên (rank cao: đoạn [1], đoạn [2]) trong [TÀI LIỆU TRUY XUẤT] hay không.
   - 1.00: Đoạn luật cốt lõi nhất nằm ngay ở đoạn [1] hoặc [2].
   - Trừ điểm: Đoạn hữu ích bị trôi xuống cuối (đoạn [4], [5]) hoặc tài liệu truy xuất chứa nhiều nhiễu không liên quan.

4. CONTEXT_RECALL (Độ bao phủ của ngữ cảnh - Thang điểm 0.00 đến 1.00):
   - Các luận điểm pháp lý cốt lõi trong [ĐÁP ÁN CHUẨN] có xuất hiện đầy đủ trong các đoạn [TÀI LIỆU TRUY XUẤT] hay không?
   - 1.00: Toàn bộ thông tin cần thiết trong đáp án chuẩn đều đã được hệ thống truy xuất thành công.
   - Trừ điểm: Thiếu các căn cứ pháp lý quan trọng được nêu trong đáp án chuẩn.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Bạn PHẢI trả về duy nhất một khối mã JSON hợp lệ (không kèm theo bất kỳ lời chào hay văn bản giải thích ngoài JSON) theo đúng cấu trúc sau:
{
  "faithfulness_score": <float từ 0.00 đến 1.00>,
  "faithfulness_reason": "<giải thích ngắn gọn 1-2 câu>",
  "answer_relevancy_score": <float từ 0.00 đến 1.00>,
  "answer_relevancy_reason": "<giải thích ngắn gọn 1-2 câu>",
  "context_precision_score": <float từ 0.00 đến 1.00>,
  "context_precision_reason": "<giải thích ngắn gọn 1-2 câu>",
  "context_recall_score": <float từ 0.00 đến 1.00>,
  "context_recall_reason": "<giải thích ngắn gọn 1-2 câu>"
}
"""


def _clean_json_response(raw_text: str) -> dict:
    """Bóc tách JSON an toàn từ kết quả sinh của LLM."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Tìm dấu ngoặc nhọn đầu tiên và cuối cùng
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]

    data = json.loads(text)
    return data


def evaluate_single_sample_llm_judge(
    llm: Any,
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
    reference: str,
) -> dict:
    """
    Chấm điểm 1 mẫu bằng Single-Pass LLM-as-a-Judge.
    Nếu có lỗi hoặc không thể chấm điểm -> Trả về 0.0 (tuyệt đối không dùng điểm giả).
    """
    # Định dạng các đoạn ngữ cảnh trích xuất
    if isinstance(retrieved_contexts, list):
        ctx_str = "\n\n".join([f"--- Đoạn [{i+1}] ---\n{c}" for i, c in enumerate(retrieved_contexts)])
    else:
        ctx_str = str(retrieved_contexts)

    user_prompt = f"""[CÂU HỎI NGƯỜI DÙNG]:
{user_input}

[TÀI LIỆU TRUY XUẤT (RETRIEVED CONTEXTS)]:
{ctx_str}

[CÂU TRẢ LỜI CỦA RAG]:
{response}

[ĐÁP ÁN CHUẨN (REFERENCE)]:
{reference}

Hãy đánh giá và xuất ra JSON theo đúng schema quy định."""

    try:
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        res = llm.invoke(messages)
        res_text = res.content if hasattr(res, "content") else str(res)
        parsed = _clean_json_response(res_text)

        f_score = max(0.0, min(1.0, float(parsed.get("faithfulness_score", 0.0))))
        ar_score = max(0.0, min(1.0, float(parsed.get("answer_relevancy_score", 0.0))))
        cp_score = max(0.0, min(1.0, float(parsed.get("context_precision_score", 0.0))))
        cr_score = max(0.0, min(1.0, float(parsed.get("context_recall_score", 0.0))))

        return {
            "faithfulness": f_score,
            "faithfulness_reason": str(parsed.get("faithfulness_reason", "")),
            "answer_relevancy": ar_score,
            "answer_relevancy_reason": str(parsed.get("answer_relevancy_reason", "")),
            "context_precision": cp_score,
            "context_precision_reason": str(parsed.get("context_precision_reason", "")),
            "context_recall": cr_score,
            "context_recall_reason": str(parsed.get("context_recall_reason", "")),
        }
    except Exception as e:
        # Khi xảy ra lỗi API hoặc parse -> Ghi nhận 0.0 tuyệt đối (không gán điểm ảo)
        return {
            "faithfulness": 0.0,
            "faithfulness_reason": f"Lỗi chấm điểm: {e}",
            "answer_relevancy": 0.0,
            "answer_relevancy_reason": f"Lỗi chấm điểm: {e}",
            "context_precision": 0.0,
            "context_precision_reason": f"Lỗi chấm điểm: {e}",
            "context_recall": 0.0,
            "context_recall_reason": f"Lỗi chấm điểm: {e}",
        }


def evaluate_llm_judge(eval_df: pd.DataFrame) -> dict:
    """
    Thực thi đánh giá toàn bộ tập dữ liệu bằng Single-Pass LLM-as-a-Judge.
    Đầu ra tương thích hoàn toàn với định dạng báo cáo và biểu đồ của hệ thống.
    """
    judge_llm = get_evaluator_llm()
    n_samples = len(eval_df)
    print(f"[LLM-Judge] Khởi động Single-Pass LLM-as-a-Judge trên {n_samples} câu hỏi...")

    judged_rows = []
    for idx, row in eval_df.iterrows():
        q = str(row.get("user_input", ""))
        resp = str(row.get("response", ""))
        retrieved = row.get("retrieved_contexts", [])
        ref = str(row.get("reference", ""))

        print(f"  [Judge {len(judged_rows)+1:02d}/{n_samples}] Đang đánh giá: {q[:60]}...")
        judge_res = evaluate_single_sample_llm_judge(
            llm=judge_llm,
            user_input=q,
            response=resp,
            retrieved_contexts=retrieved,
            reference=ref,
        )

        row_dict = dict(row)
        row_dict.update(judge_res)
        judged_rows.append(row_dict)

    results_df = pd.DataFrame(judged_rows)

    # Tính điểm trung bình (bỏ qua NaN nếu có, mặc định nếu không có điểm là 0.0)
    scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in results_df.columns:
            val = results_df[metric].dropna().mean()
            scores[metric] = round(float(val), 4) if pd.notna(val) else 0.0
        else:
            scores[metric] = 0.0

    return {
        "scores": scores,
        "results_df": results_df,
    }
