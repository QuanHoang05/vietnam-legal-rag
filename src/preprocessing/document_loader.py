"""
Bộ nạp tài liệu thông minh đa định dạng (PDF và DOCX) cho toàn bộ thư mục tri thức.
Hỗ trợ tự động phân loại danh mục (category) theo cấu trúc thư mục (AI, LuatDatDai, LuatLaoDong).
Gán metadata phân cấp đầy đủ phục vụ cấu trúc cây tri thức (Tree Structure).
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from langchain_core.documents import Document
from .text_cleaner import clean_vietnamese_text


class UniversalDocumentLoader:
    """
    Bộ nạp tài liệu tự động nhận diện cả .pdf và .docx.
    Trích xuất cấu trúc văn bản kèm Metadata cây: category, source, file_type, section_id.
    """

    def load_pdf(self, pdf_file: Path, category: str = "General", max_pages: Optional[int] = None) -> List[Document]:
        """Đọc và làm sạch nội dung file PDF hiệu năng cao. Không giới hạn số trang (max_pages=None)."""
        docs = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_file))
            total_pages = len(reader.pages)
            limit = min(total_pages, max_pages) if max_pages else total_pages
            if max_pages and total_pages > max_pages:
                print(f"  ⚠️  [UniversalLoader] CẮT TRANG: '{pdf_file.name}' có {total_pages} trang, chỉ nạp {max_pages} trang đầu. Tăng max_pages để nạp đủ.")
            for i in range(limit):
                try:
                    text = reader.pages[i].extract_text() or ""
                    cleaned = clean_vietnamese_text(text)
                    if cleaned.strip():
                        docs.append(Document(
                            page_content=cleaned,
                            metadata={
                                "source": pdf_file.name,
                                "category": category,
                                "file_type": "pdf",
                                "page": i + 1,
                                "total_pages": total_pages,
                                "tree_path": f"{category}/{pdf_file.stem}/page_{i + 1}",
                            }
                        ))
                except Exception:
                    continue
        except Exception as e:
            print(f"[UniversalLoader] Không thể đọc PDF {pdf_file.name}: {e}")
            return []

        return docs


    def load_docx(self, docx_file: Path, category: str = "General") -> List[Document]:
        """
        Đọc nội dung văn bản Word (.docx) bằng native zipfile + XML.
        Không yêu cầu cài thêm thư viện phụ thuộc, bảo đảm tương thích 100%.
        Tự động nhận diện cấu trúc Chương / Điều / Mục cho văn bản pháp luật.
        """
        docs = []
        try:
            with zipfile.ZipFile(docx_file) as z:
                xml_content = z.read("word/document.xml")
            
            tree = ET.fromstring(xml_content)
            # Namespace WordprocessingML
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            
            paragraphs = []
            for p in tree.iter(f"{{{ns['w']}}}p"):
                texts = [node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text]
                if texts:
                    p_text = "".join(texts).strip()
                    if p_text:
                        paragraphs.append(p_text)

            # Nạp toàn bộ các đoạn văn đã làm sạch, bảo toàn cấu trúc chuẩn cho LegalArticleChunker
            clean_paragraphs = []
            for p in paragraphs:
                p_clean = clean_vietnamese_text(p)
                if p_clean:
                    clean_paragraphs.append(p_clean)

            if clean_paragraphs:
                full_text = "\n".join(clean_paragraphs)
                docs.append(Document(
                    page_content=full_text,
                    metadata={
                        "source": docx_file.name,
                        "category": category,
                        "file_type": "docx",
                        "tree_path": f"{category}/{docx_file.stem}",
                    }
                ))

        except Exception as e:
            print(f"[UniversalLoader] Lỗi đọc DOCX {docx_file.name}: {e}")

        return docs

    def load_all(self, base_dir: str, _max_files_per_category: Optional[int] = None) -> List[Document]:
        """
        Duyệt đệ quy toàn bộ thư mục tri thức, nạp tất cả PDF và DOCX.
        Tự động phân nhóm Category theo tên thư mục cha.
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {base_dir}")

        all_docs: List[Document] = []
        found_files = []

        # Quét mọi file .pdf và .docx
        for ext in ["*.pdf", "*.docx"]:
            found_files.extend(list(base_path.glob(ext)) + list(base_path.glob(f"**/{ext}")))

        # Loại trùng
        unique_files = sorted(list(set(found_files)), key=lambda p: str(p))

        print(f"[UniversalLoader] Tìm thấy tổng cộng {len(unique_files)} tài liệu trong '{base_dir}'.")

        for file_path in tqdm(unique_files, desc="Đang nạp kho tài liệu (PDF & DOCX)"):
            # Lấy Category từ thư mục con (ví dụ: ducument/LuatDatDai -> LuatDatDai)
            rel = file_path.relative_to(base_path)
            category = rel.parts[0] if len(rel.parts) > 1 else "Root"

            if file_path.suffix.lower() == ".pdf":
                docs = self.load_pdf(file_path, category=category)
                all_docs.extend(docs)
            elif file_path.suffix.lower() == ".docx":
                docs = self.load_docx(file_path, category=category)
                all_docs.extend(docs)

        print(f"[UniversalLoader] Đã tải hoàn tất {len(all_docs)} đoạn tài liệu gốc với đầy đủ Metadata cây.\n")
        return all_docs
