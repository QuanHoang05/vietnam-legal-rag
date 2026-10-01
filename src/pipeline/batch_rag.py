"""
Pipeline BatchRAG cơ bản và mở rộng hỗ trợ Re-ranking.
Quản lý luồng xử lý truy vấn theo lô, kết hợp Retriever và LLM.
(Theo mục VII.4.4 & VII.5.4 trong tài liệu Insight into RAG)
"""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from configs.settings import settings
from .answer_parser import FocusedAnswerParser


class BatchRAG:
    """
    Điều phối luồng thực thi RAG theo batch để tối ưu hóa băng thông API và giảm độ trễ.
    Hỗ trợ tích hợp tùy chọn Cross-Encoder Reranker.
    """

    def __init__(
        self,
        llm: Any,
        reranker: Optional[Any] = None,
        batch_size: int = settings.BATCH_SIZE,
    ):
        self.llm = llm
        self.reranker = reranker
        self.batch_size = batch_size

        self.prompt_template = """Bạn là chuyên gia phân tích tài liệu tiếng Việt. NHIỆM VỤ CỦA BẠN: Chỉ trả lời câu hỏi dựa trên DUY NHẤT các tài liệu tham khảo được cung cấp bên dưới. 

LUẬT LỆ NGHIÊM NGẶT:
1. KHÔNG SỬ DỤNG kiến thức bên ngoài, không tự suy diễn hoặc bịa đặt thông tin.
2. Nếu tài liệu tham khảo KHÔNG chứa đủ thông tin để trả lời câu hỏi, hãy trả lời chính xác là "Không có thông tin".
3. Câu trả lời phải ngắn gọn, súc tích và đi thẳng vào trọng tâm.

{history_block}[TÀI LIỆU THAM KHẢO]:
{context}

[CÂU HỎI HIỆN TẠI]:
{question}

[TRẢ LỜI]:"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.answer_parser = FocusedAnswerParser()

    def _format_docs(self, docs: List[Any]) -> str:
        """Ghép nối các đoạn văn bản thành ngữ cảnh duy nhất. Tận dụng parent_snippet nếu có để mở rộng bối cảnh suy luận cho LLM."""
        formatted = []
        seen = set()
        for doc in docs:
            content = (doc.page_content or "").strip()
            # Mở rộng bằng parent_snippet nếu chunk con có liên kết cây cha
            parent_snippet = doc.metadata.get("parent_snippet") if hasattr(doc, "metadata") and doc.metadata else None
            text_to_use = parent_snippet if (parent_snippet and len(parent_snippet) > len(content)) else content
            if text_to_use and len(text_to_use) > 30 and text_to_use not in seen:
                formatted.append(text_to_use)
                seen.add(text_to_use)
        return "\n\n".join(formatted)

    def batch_retrieve(self, questions: List[str], retriever: Any) -> List[Dict[str, Any]]:
        """Truy xuất tài liệu cho danh sách câu hỏi và áp dụng Reranking nếu có"""
        desc = "Retrieving & reranking documents" if self.reranker else "Retrieving documents"
        all_contexts = []

        for question in tqdm(questions, desc=desc):
            docs = retriever.invoke(question)

            # Áp dụng Re-ranking nếu được cấu hình
            if self.reranker is not None and hasattr(self.reranker, "rerank"):
                docs = self.reranker.rerank(question, docs)

            contexts = [(doc.page_content or "") for doc in docs]
            formatted_context = self._format_docs(docs)

            all_contexts.append({
                "question": question,
                "contexts": contexts,
                "documents": docs,
                "formatted_context": formatted_context,
            })
        return all_contexts

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Gọi LLM sinh câu trả lời theo từng batch để tối ưu tốc độ"""
        all_answers = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating answers"):
            batch = prompts[i: i + self.batch_size]
            messages_batch = [[HumanMessage(content=p)] for p in batch]

            try:
                batch_results = self.llm.batch(messages_batch)
                for result in batch_results:
                    content = result.content if hasattr(result, "content") else str(result)
                    answer = self.answer_parser.parse(content)
                    all_answers.append(answer)
            except Exception as e:
                # Fallback gọi tuần tự nếu batch gặp lỗi (rate limit, v.v.)
                print(f"[BatchRAG] Lỗi batch {i}: {e}. Chuyển sang gọi tuần tự...")
                for p in batch:
                    try:
                        res = self.llm.invoke([HumanMessage(content=p)])
                        c = res.content if hasattr(res, "content") else str(res)
                        all_answers.append(self.answer_parser.parse(c))
                    except Exception as err:
                        print(f"[BatchRAG] Lỗi tuần tự: {err}")
                        all_answers.append("Không có thông tin")

        return all_answers

    def answer_with_contexts_batch(self, questions: List[str], retriever: Any, history_text: str = "") -> List[Dict[str, Any]]:
        """Quy trình trọn vẹn: Retrieve ngữ cảnh -> Sinh câu trả lời -> Đóng gói kết quả"""
        retrieved_data = self.batch_retrieve(questions, retriever)
        history_block = ""
        if history_text:
            history_block = f"[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n{history_text}\n\n"
        prompts = [
            self.prompt.format(
                history_block=history_block,
                context=data["formatted_context"],
                question=data["question"]
            )
            for data in retrieved_data
        ]
        answers = self.batch_generate(prompts)

        results = []
        for data, answer in zip(retrieved_data, answers):
            results.append({
                "answer": answer,
                "contexts": data["contexts"],
                "documents": data.get("documents", []),
            })
        return results

