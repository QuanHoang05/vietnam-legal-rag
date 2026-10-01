"""
Bộ giải mã và làm sạch câu trả lời sinh ra từ LLM.
Loại bỏ các cụm từ dẫn dắt dư thừa ("Dựa vào tài liệu...", "Câu trả lời là...")
để phục vụ quá trình chấm điểm tự động chính xác nhất.
(Theo mục VII.4.4 trong tài liệu Insight into RAG)
"""

import re
from langchain_core.output_parsers import StrOutputParser


class FocusedAnswerParser(StrOutputParser):
    """
    Hậu xử lý kết quả: loại bỏ tiền tố [TRẢ LỜI]:, bullet points,
    và khoảng trắng dư thừa.
    """

    def parse(self, text: str) -> str:
        text = (text or "").strip()
        if "[TRẢ LỜI]:" in text:
            answer = text.split("[TRẢ LỜI]:")[-1].strip()
        elif "[ANSWER]:" in text:
            answer = text.split("[ANSWER]:")[-1].strip()
        else:
            answer = text

        # Xóa các ký tự đầu dòng kiểu bullet: •, -, *
        answer = re.sub(r"^\s*[\u2022\-\*]\s*", "", answer, flags=re.MULTILINE)
        # Rút gọn nhiều dấu xuống dòng thành dấu cách
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer
