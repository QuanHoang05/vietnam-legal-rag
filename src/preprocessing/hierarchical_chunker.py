"""
Hierarchical / Tree-based Chunking (Cắt đoạn theo cấu trúc cây tri thức).
Kết hợp mô hình Parent-Child Chunking:
- Child Chunk (Lá cây, ~350 ký tự): Tối ưu độ chính xác của Vector Search (không bị loãng ngữ nghĩa).
- Parent Chunk (Nhánh cây, ~1000 ký tự): Cung cấp bối cảnh trọn vẹn, phong phú cho LLM suy luận.
- Định danh SHA-256 xác định (Deterministic IDs): Chống nhân bản Vector DB.
"""

import hashlib
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HierarchicalTreeChunker:
    """
    Bộ phân đoạn tài liệu theo cấu trúc cây (Tree-structured Hierarchical Chunking).
    Mỗi đoạn con lưu liên kết trỏ về đoạn cha (Parent Document Linkage).
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 128,
        child_chunk_size: int = 350,
        child_chunk_overlap: int = 60,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

    def _generate_doc_id(self, text: str, meta: Dict[str, Any], suffix: str = "") -> str:
        """Sinh mã băm SHA-256 cố định đảm bảo tính duy nhất và tái lập (P1-2)"""
        raw_key = f"{meta.get('category', '')}_{meta.get('source', '')}_{meta.get('tree_path', '')}_{text[:60]}_{suffix}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def split(self, raw_documents: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
        """
        Phân tách danh sách Document thành các Child Chunk kèm Metadata cây và ánh xạ Parent Chunk.
        
        Returns:
            child_docs: Danh sách Document đoạn con đưa vào Vector Store.
            parent_store: Từ điển ánh xạ {parent_id: parent_full_content} để phục vụ trích xuất đầy đủ.
        """
        child_docs: List[Document] = []
        parent_store: Dict[str, str] = {}

        # 1. Cắt thành Parent Chunks (Nhánh cây)
        parent_chunks = self.parent_splitter.split_documents(raw_documents)

        for p_idx, p_doc in enumerate(parent_chunks):
            p_meta = dict(p_doc.metadata)
            parent_id = f"p_{self._generate_doc_id(p_doc.page_content, p_meta, str(p_idx))}"
            parent_store[parent_id] = p_doc.page_content

            # 2. Cắt tiếp Parent Chunk thành các Child Chunks (Lá cây)
            child_sub_chunks = self.child_splitter.split_text(p_doc.page_content)

            for c_idx, c_text in enumerate(child_sub_chunks):
                clean_text = c_text.strip()
                if not clean_text or len(clean_text) < 20:
                    continue

                child_id = f"c_{parent_id}_{c_idx}"
                c_meta = dict(p_meta)
                c_meta["chunk_id"] = child_id
                c_meta["parent_id"] = parent_id
                c_meta["parent_snippet"] = p_doc.page_content.strip()
                c_meta["level"] = "child_leaf"
                c_meta["tree_path"] = f"{p_meta.get('tree_path', 'root')}/chunk_{c_idx}"
                c_meta["char_length"] = len(clean_text)

                child_doc = Document(
                    page_content=clean_text,
                    metadata=c_meta,
                )
                child_docs.append(child_doc)

        print(f"[HierarchicalTreeChunker] Đã tạo {len(parent_store)} Parent Chunks và {len(child_docs)} Child Chunks (Tree Leaves).")
        return child_docs, parent_store
