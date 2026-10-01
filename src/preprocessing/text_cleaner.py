"""
Tiền xử lý và làm sạch văn bản tiếng Việt.
Chuẩn hóa bảng mã Unicode NFC, loại bỏ ký tự điều khiển lạ từ PDF.
"""

import re
import unicodedata


def clean_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt theo chuẩn NFC và loại bỏ ký tự nhiễu.
    (Theo mục VII.4.3 trong tài liệu Insight into RAG)
    """
    if not text:
        return ""
    # Chuẩn hóa về NFC
    text = unicodedata.normalize("NFC", text)
    # Loại bỏ các ký tự điều khiển không mong muốn (Category 'C') ngoại trừ newline và tab
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("C") or char in "\n\t"
    )
    # Loại bỏ khoảng trắng thừa liên tiếp
    text = re.sub(r"[ \t]+", " ", text)
    # Nối dòng khi dòng trước không kết thúc bằng dấu kết câu (tránh đứt từ như "mô\nhình")
    text = re.sub(r"(?<![.!?;:\n])\n(?=[a-zà-ỹ0-9])", " ", text)
    # Rút gọn các dòng trống liên tiếp
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

