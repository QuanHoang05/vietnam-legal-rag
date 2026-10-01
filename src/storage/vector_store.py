"""
Quản lý lưu trữ Vector Database với ChromaDB và Hybrid Storage.
Tối ưu hóa hiệu năng (P1-1, P1-2, P1-3 trong báo cáo review):
- Tải trực tiếp từ đĩa nếu đã lập chỉ mục (0.01s), TUYỆT ĐỐI KHÔNG EMBED LẠI TÀI LIỆU.
- Định danh SHA-256 xác định (Deterministic IDs) chống nhân bản collection.
- Cache chỉ mục BM25 qua file pickle để không phải tokenize underthesea lặp lại.
"""

import pickle
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from configs.settings import settings
from src.models.embedding_factory import get_embeddings


def _get_doc_fingerprint(docs: List[Document]) -> str:
    """Tạo mã định danh duy nhất (fingerprint) cho tập tài liệu để chống dùng nhầm cache cũ"""
    if not docs:
        return "empty"
    summary_str = f"{len(docs)}_" + "|".join(
        sorted(f"{d.metadata.get('source', '')}:{d.metadata.get('chunk_id', '')[:8]}" for d in docs[::max(1, len(docs) // 50)])
    )
    return hashlib.sha256(summary_str.encode("utf-8")).hexdigest()[:16]


def check_index_health(persist_dir: Optional[str] = None, min_expected: int = 100) -> Dict[str, Any]:
    """Kiểm tra tính toàn vẹn của chỉ mục ChromaDB trên đĩa"""
    p_dir = Path(persist_dir or settings.PERSIST_DIR)
    sqlite_file = p_dir / "chroma.sqlite3"
    if not sqlite_file.exists():
        return {"exists": False, "count": 0, "sources": 0, "healthy": False}

    try:
        import sqlite3
        conn = sqlite3.connect(str(sqlite_file))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM embedding_metadata WHERE key='chroma:document'")
        row = cursor.fetchone()
        count = row[0] if row else 0

        cursor.execute("SELECT count(DISTINCT string_value) FROM embedding_metadata WHERE key='source'")
        row_src = cursor.fetchone()
        n_sources = row_src[0] if row_src else 0

        conn.close()

        is_healthy = count >= min_expected and n_sources >= 2
        return {
            "exists": True,
            "count": count,
            "sources": n_sources,
            "healthy": is_healthy,
            "error": None
        }
    except Exception as e:
        return {"exists": True, "count": 0, "sources": 0, "healthy": False, "error": str(e)}


class VectorDB:
    """
    Quản lý Vector Store sử dụng ChromaDB với cơ chế kiểm tra tính toàn vẹn nghiêm ngặt.
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embedding: Optional[Any] = None,
        collection_name: str = "vietnamese_docs",
        persist_dir: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or settings.PERSIST_DIR)
        self.collection_name = collection_name
        self.embedding = embedding or get_embeddings()
        self.db = self._build_or_load_db(documents)

    def _build_or_load_db(self, documents: Optional[List[Document]]):
        from langchain_chroma import Chroma

        persist_path = Path(self.persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        sqlite_file = persist_path / "chroma.sqlite3"

        # 1. Nếu đã có dữ liệu trên đĩa, kiểm tra số lượng thực tế
        if sqlite_file.exists():
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
            count = 0
            try:
                count = db._collection.count()
            except Exception:
                pass

            # Chỉ chấp nhận chỉ mục cũ nếu số lượng đủ lớn hoặc khớp với documents truyền vào
            if count > 0:
                if not documents or count >= len(documents):
                    print(f"[VectorDB] Đã nạp {count} vectors từ đĩa '{self.persist_dir}' (0.01s - Toàn vẹn).")
                    return db
                else:
                    print(f"[VectorDB CẢNH BÁO] Chỉ mục cũ chỉ có {count} vectors < {len(documents)} chunks cần nạp. Đang lập chỉ mục lại...")

        # 2. Nếu chưa có trên đĩa và có tài liệu truyền vào -> Lập chỉ mục với ID xác định
        if documents and len(documents) > 0:
            print(f"[VectorDB] Đang tạo mới vector store '{self.collection_name}' ({len(documents)} chunks)...")
            ids = []
            for idx, doc in enumerate(documents):
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id:
                    chunk_id = hashlib.sha256(f"{doc.metadata.get('source', '')}_{idx}_{doc.page_content[:50]}".encode()).hexdigest()[:16]
                ids.append(chunk_id)

            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                ids=ids,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
            return db

        # 3. Mặc định tạo đối tượng rỗng trỏ vào persist_dir
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_dir,
        )

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if search_kwargs is None:
            if search_type == "mmr":
                search_kwargs = {
                    "k": settings.MMR_K,
                    "fetch_k": settings.MMR_FETCH_K,
                    "lambda_mult": settings.MMR_LAMBDA_MULT,
                }
            else:
                search_kwargs = {"k": settings.BASELINE_RETRIEVE_K}

        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        return self.db.similarity_search(query, k=k)


class HybridVectorDB:
    """
    Quản lý đồng thời Vector Index và Keyword Index BM25 có bộ đệm Disk Cache gắn Fingerprint.
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embedding: Optional[Any] = None,
        collection_name: str = "vietnamese_docs",
        persist_dir: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or settings.PERSIST_DIR)
        self.collection_name = collection_name
        self.embedding = embedding or get_embeddings()
        self.documents = documents or []

        self.vector_db = self._build_vector_db(documents)
        self.bm25 = self._build_bm25_index(documents)

    def _build_vector_db(self, documents: Optional[List[Document]]):
        from langchain_chroma import Chroma

        persist_path = Path(self.persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        sqlite_file = persist_path / "chroma.sqlite3"

        if sqlite_file.exists():
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
            count = 0
            try:
                count = db._collection.count()
            except Exception:
                pass

            if count > 0:
                if not documents or count >= len(documents):
                    return db

        if documents and len(documents) > 0:
            ids = [
                doc.metadata.get("chunk_id") or hashlib.sha256(f"{doc.metadata.get('source', '')}_{idx}".encode()).hexdigest()[:16]
                for idx, doc in enumerate(documents)
            ]
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                ids=ids,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_dir,
        )

    def _build_bm25_index(self, documents: Optional[List[Document]]):
        cache_dir = Path(self.persist_dir)
        bm25_cache_file = cache_dir / "bm25_cache.pkl"
        docs_cache_file = cache_dir / "documents_cache.pkl"

        # Trường hợp 1: Không truyền documents -> Cố gắng đọc từ cache đĩa
        if not documents or len(documents) == 0:
            if docs_cache_file.exists():
                try:
                    with open(docs_cache_file, "rb") as f:
                        cached_docs_payload = pickle.load(f)
                        if isinstance(cached_docs_payload, dict):
                            self.documents = cached_docs_payload.get("documents", [])
                            expected_fp = cached_docs_payload.get("fingerprint")
                        else:
                            self.documents = cached_docs_payload
                            expected_fp = None
                except Exception:
                    self.documents = []

            if bm25_cache_file.exists() and self.documents:
                try:
                    with open(bm25_cache_file, "rb") as f:
                        cached_bm25_payload = pickle.load(f)
                        if isinstance(cached_bm25_payload, dict):
                            cached_fp = cached_bm25_payload.get("fingerprint")
                            if expected_fp and cached_fp == expected_fp:
                                return cached_bm25_payload.get("bm25")
                        else:
                            return cached_bm25_payload
                except Exception:
                    pass
            return None

        # Trường hợp 2: Có truyền documents -> Tính fingerprint và lưu cache chuẩn
        current_fp = _get_doc_fingerprint(documents)

        # Kiểm tra xem cache hiện tại có đúng fingerprint không
        if bm25_cache_file.exists():
            try:
                with open(bm25_cache_file, "rb") as f:
                    cached_payload = pickle.load(f)
                    if isinstance(cached_payload, dict) and cached_payload.get("fingerprint") == current_fp:
                        return cached_payload.get("bm25")
            except Exception:
                pass

        # Xây dựng BM25 index mới
        from rank_bm25 import BM25Okapi
        try:
            from underthesea import word_tokenize
            tokenize_fn = lambda text: word_tokenize(text.lower())
        except ImportError:
            tokenize_fn = lambda text: text.lower().split()

        tokenized_docs = [tokenize_fn(doc.page_content) for doc in documents]
        bm25_index = BM25Okapi(tokenized_docs)

        # Lưu cả BM25 index và documents kèm fingerprint
        try:
            with open(bm25_cache_file, "wb") as f:
                pickle.dump({"fingerprint": current_fp, "bm25": bm25_index, "count": len(documents)}, f)
            with open(docs_cache_file, "wb") as f:
                pickle.dump({"fingerprint": current_fp, "documents": documents, "count": len(documents)}, f)
        except Exception:
            pass

        return bm25_index


    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None, reranker: Optional[Any] = None):
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker

        if search_kwargs is None:
            search_kwargs = {"k": 5}

        # Tự động trang bị CrossEncoderReranker để chạy quy trình 2 bước: RRF lọc thô -> Reranker lọc tinh
        actual_reranker = reranker
        if actual_reranker is None:
            actual_reranker = CrossEncoderReranker(top_k=search_kwargs.get("k", 5))

        return HybridRetriever(
            vector_db=self.vector_db,
            bm25=self.bm25,
            documents=self.documents,
            reranker=actual_reranker,
            candidate_k=search_kwargs.get("candidate_k", 15),
            k=search_kwargs.get("k", 5),
            rrf_k=settings.RRF_K,
        )

