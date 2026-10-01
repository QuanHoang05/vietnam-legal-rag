"""
Legal Structure Chunker (Bộ phân đoạn chuyên sâu cho Văn bản Quy phạm Pháp luật Việt Nam).
Tối ưu hóa đặc thù cho các bộ luật (Luật Đất Đai, Bộ Luật Lao Động, v.v.):
1. Phân tầng ngữ nghĩa pháp lý nghiêm ngặt: Chương -> Mục -> Điều -> Khoản -> Điểm.
2. Bảo toàn tính toàn vẹn của Điều luật: Một Điều là một đơn vị ngữ nghĩa nguyên vẹn (Atomic Unit).
3. Chia nhỏ có điều kiện (Conditional Clause Splitting): Chỉ chia theo Khoản khi Điều quá dài (>1200-1500 ký tự).
4. Tiêm ngữ cảnh pháp lý (Breadcrumb Context Injection): Mọi chunk con đều có header chứa:
   [Tên Luật | Tên Chương | Điều X. Tên Điều (Khoản Y)]
5. Định danh SHA-256 xác định (Deterministic Chunk ID): Chống trùng lặp trong Vector DB.
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from langchain_core.documents import Document
from configs.settings import settings


class LegalArticleChunker:
    """
    Bộ chia đoạn cấu trúc văn bản pháp luật Việt Nam.
    """

    def __init__(self, max_chunk_size: Optional[int] = None, min_chunk_size: Optional[int] = None):
        self.max_chunk_size = max_chunk_size if max_chunk_size is not None else settings.LEGAL_CHUNK_MAX_SIZE
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else settings.LEGAL_CHUNK_MIN_SIZE
        
        # Regex nhận diện các mốc cấu trúc pháp luật
        self.re_chapter = re.compile(r"^(Chương|CHƯƠNG)\s+([IVXLCDM\d]+)[\.:\s]*(.*)$", re.IGNORECASE)
        self.re_article = re.compile(r"^(Điều|ĐIỀU)\s+(\d+)[\.:\s]*(.*)$", re.IGNORECASE)
        self.re_clause = re.compile(r"^(\d+)[\.\s]+(.*)$")

    def _detect_law_title(self, source_name: str = "", category: str = "") -> str:
        """Tự động suy luận tên văn bản luật từ nội dung hoặc tên file/danh mục."""
        source_upper = source_name.upper()
        cat_upper = category.upper()

        if "31_2024" in source_upper or "DATDAI" in cat_upper or "ĐẤT ĐAI" in source_upper:
            return "Luật Đất đai 2024 (Luật số 31/2024/QH15)"
        if "45_2019" in source_upper or "LAODONG" in cat_upper or "LAO ĐỘNG" in source_upper:
            return "Bộ luật Lao động 2019 (Bộ luật số 45/2019/QH14)"

        return f"Văn bản pháp luật {source_name}" if source_name else "Văn bản pháp luật"

    def split(self, raw_documents: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
        """
        Phân tách danh sách Document thô thành các Chunk pháp lý chuẩn mực.
        Tương thích chữ ký hàm với pipeline lập chỉ mục (child_docs, parent_store).
        
        Returns:
            legal_chunks: Danh sách Document phân đoạn theo Điều/Khoản kèm breadcrumbs.
            parent_store: Từ điển ánh xạ {article_id: full_article_text} phục vụ mở rộng ngữ cảnh.
        """
        all_chunks: List[Document] = []
        parent_store: Dict[str, str] = {}

        # Gom nhóm Document theo nguồn tài liệu (file)
        docs_by_source: Dict[str, List[Document]] = {}
        for d in raw_documents:
            src = d.metadata.get("source", "unknown")
            docs_by_source.setdefault(src, []).append(d)

        for source_name, src_docs in docs_by_source.items():
            category = src_docs[0].metadata.get("category", "Legal")
            law_title = self._detect_law_title(source_name=source_name, category=category)

            # Thu thập toàn bộ các dòng văn bản theo thứ tự
            all_lines: List[str] = []
            for doc in src_docs:
                lines = doc.page_content.splitlines()
                for l in lines:
                    cl = l.strip()
                    if cl:
                        all_lines.append(cl)

            # Phân tách cấu trúc
            chunks_for_file, file_parent_store = self._parse_lines_to_legal_chunks(
                lines=all_lines,
                source_name=source_name,
                category=category,
                law_title=law_title
            )
            all_chunks.extend(chunks_for_file)
            parent_store.update(file_parent_store)

        print(f"[LegalArticleChunker] Đã tạo {len(all_chunks)} chunks pháp lý chuẩn Điều/Khoản từ {len(raw_documents)} raw docs.")
        return all_chunks, parent_store

    def _parse_lines_to_legal_chunks(
        self,
        lines: List[str],
        source_name: str,
        category: str,
        law_title: str
    ) -> Tuple[List[Document], Dict[str, str]]:
        chunks: List[Document] = []
        parent_store: Dict[str, str] = {}

        current_chapter = "Quy định chung"
        current_article_num = ""
        current_article_title = ""
        current_preamble: List[str] = []
        current_clauses: List[Dict[str, Any]] = []
        in_article = False

        def flush_article():
            nonlocal current_article_num, current_article_title, current_preamble, current_clauses
            if not current_article_num:
                return

            full_article_heading = f"{current_article_num}. {current_article_title}".strip()
            preamble_text = "\n".join(current_preamble).strip()

            all_parts = []
            if preamble_text:
                all_parts.append(preamble_text)
            for cl in current_clauses:
                all_parts.append(cl["text"])
            full_article_body = "\n".join(all_parts)

            # Lưu vào parent_store để LLM có thể đọc trọn vẹn toàn bộ Điều khi cần
            article_key = f"{category}_{current_article_num}"
            full_article_with_header = f"[{law_title} | {current_chapter} | {full_article_heading}]\n{full_article_body}"
            parent_store[article_key] = full_article_with_header

            base_header = f"[{law_title} | {current_chapter} | {full_article_heading}]"

            # TRƯỜNG HỢP 1: Điều ngắn hoặc vừa phải (<= max_chunk_size) -> 1 Chunk hoàn chỉnh
            if (len(base_header) + len(full_article_body) + 2 <= self.max_chunk_size) or len(current_clauses) <= 1:
                content = f"{base_header}\n{full_article_body}".strip() if full_article_body else f"{base_header}\n{full_article_heading}"
                raw_hash = f"{law_title}_{current_article_num}_full"
                c_id = f"legal_{hashlib.sha256(raw_hash.encode('utf-8')).hexdigest()[:16]}"

                chunks.append(Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "category": category,
                        "law_title": law_title,
                        "chapter": current_chapter,
                        "article": current_article_num,
                        "article_title": current_article_title,
                        "clause_scope": "Toàn văn điều",
                        "chunk_id": c_id,
                        "parent_id": article_key,
                        "parent_snippet": full_article_with_header,
                        "char_length": len(content)
                    }
                ))
            else:
                # TRƯỜNG HỢP 2: Điều dài (> max_chunk_size) -> Gom nhóm các Khoản thành từng Sub-chunk
                batch_clauses: List[Dict[str, Any]] = []
                batch_len = 0
                batch_idx = 1

                for cl in current_clauses:
                    cl_text = cl["text"]
                    # Nếu vượt quá dung lượng cho phép thì chốt batch hiện tại
                    if batch_clauses and (batch_len + len(cl_text) > self.max_chunk_size - len(base_header) - 150):
                        c_start = batch_clauses[0]["num"]
                        c_end = batch_clauses[-1]["num"]
                        scope_str = f"Khoản {c_start} - {c_end}" if c_start != c_end else f"Khoản {c_start}"

                        sub_header = f"[{law_title} | {current_chapter} | {full_article_heading} ({scope_str})]"
                        sub_body_list = []
                        if preamble_text and batch_idx == 1:
                            sub_body_list.append(preamble_text)
                        sub_body_list.extend([c["text"] for c in batch_clauses])
                        sub_content = f"{sub_header}\n" + "\n".join(sub_body_list)

                        raw_hash = f"{law_title}_{current_article_num}_p{batch_idx}"
                        sub_id = f"legal_{hashlib.sha256(raw_hash.encode('utf-8')).hexdigest()[:16]}"

                        chunks.append(Document(
                            page_content=sub_content,
                            metadata={
                                "source": source_name,
                                "category": category,
                                "law_title": law_title,
                                "chapter": current_chapter,
                                "article": current_article_num,
                                "article_title": current_article_title,
                                "clause_scope": scope_str,
                                "chunk_id": sub_id,
                                "parent_id": article_key,
                                "parent_snippet": full_article_with_header,
                                "char_length": len(sub_content)
                            }
                        ))

                        batch_clauses = []
                        batch_len = 0
                        batch_idx += 1

                    batch_clauses.append(cl)
                    batch_len += len(cl_text)

                # Batch còn lại cuối cùng
                if batch_clauses:
                    c_start = batch_clauses[0]["num"]
                    c_end = batch_clauses[-1]["num"]
                    scope_str = f"Khoản {c_start} - {c_end}" if c_start != c_end else f"Khoản {c_start}"

                    sub_header = f"[{law_title} | {current_chapter} | {full_article_heading} ({scope_str})]"
                    sub_body_list = []
                    if preamble_text and batch_idx == 1:
                        sub_body_list.append(preamble_text)
                    sub_body_list.extend([c["text"] for c in batch_clauses])
                    sub_content = f"{sub_header}\n" + "\n".join(sub_body_list)

                    raw_hash = f"{law_title}_{current_article_num}_p{batch_idx}"
                    sub_id = f"legal_{hashlib.sha256(raw_hash.encode('utf-8')).hexdigest()[:16]}"

                    chunks.append(Document(
                        page_content=sub_content,
                        metadata={
                            "source": source_name,
                            "category": category,
                            "law_title": law_title,
                            "chapter": current_chapter,
                            "article": current_article_num,
                            "article_title": current_article_title,
                            "clause_scope": scope_str,
                            "chunk_id": sub_id,
                            "parent_id": article_key,
                            "parent_snippet": full_article_with_header,
                            "char_length": len(sub_content)
                        }
                    ))

            # Reset cho Điều tiếp theo
            current_article_num = ""
            current_article_title = ""
            current_preamble = []
            current_clauses = []

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            # 1. Nhận diện Chương
            m_chap = self.re_chapter.match(line_clean)
            if m_chap:
                current_chapter = f"Chương {m_chap.group(2)}"
                if m_chap.group(3):
                    current_chapter += f": {m_chap.group(3)}"
                continue

            # 2. Nhận diện Điều
            m_art = self.re_article.match(line_clean)
            if m_art:
                flush_article()
                in_article = True
                current_article_num = f"Điều {m_art.group(2)}"
                current_article_title = m_art.group(3).strip()
                continue

            if in_article:
                # 3. Nhận diện Khoản: 1. ..., 2. ...
                m_clause = self.re_clause.match(line_clean)
                if m_clause and m_clause.group(1).isdigit():
                    current_clauses.append({
                        "num": m_clause.group(1),
                        "text": line_clean
                    })
                else:
                    if not current_clauses:
                        current_preamble.append(line_clean)
                    else:
                        current_clauses[-1]["text"] += "\n" + line_clean

        flush_article()
        return chunks, parent_store
