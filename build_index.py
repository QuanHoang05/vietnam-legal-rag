"""
KỊCH BẢN LẬP CHỈ MỤC TRƯỚC (PRE-INDEXING / OFFLINE INGESTION)
Giải quyết triệt để vấn đề P1-1 và P1-2 từ báo cáo kỹ thuật:
1. Đọc toàn bộ tài liệu (PDF & DOCX) trong thư mục 'ducument' (AI, Luật Đất Đai, Luật Lao Động).
2. Phân đoạn văn bản theo cấu trúc cây phân cấp (Hierarchical Tree / Parent-Child Chunking).
3. Đính kèm Metadata cây hoàn chỉnh: category, source, tree_path, section.
4. Lập chỉ mục Vector DB một lần duy nhất vào 'chroma_data' với SHA-256 ID xác định.
5. Cache chỉ mục BM25 ra đĩa (bm25_cache.pkl).
"""

import sys
import time
import shutil
import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

# Đảm bảo console Windows in ký tự tiếng Việt không lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo import được cấu hình từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.settings import settings
from src.preprocessing.document_loader import UniversalDocumentLoader
from src.preprocessing.legal_chunker import LegalArticleChunker
from src.models.embedding_factory import get_embeddings
from src.storage.vector_store import VectorDB, HybridVectorDB


def build_full_knowledge_index(
    clean: bool = False,
    max_files: Optional[int] = None,
    max_chunks: Optional[int] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    batch_size: int = 64,
):
    actual_model = "BAAI/bge-m3"
    actual_provider = "local"
    actual_key = ""

    print("=" * 70)
    print(" [*] BAT DAU QUY TRINH LAP CHI MUC KHO TRI THUC PHAP LUAT (PRE-INDEXING)")
    print(f" - Thư mục tài liệu:   {settings.DATA_DIR}")
    print(f" - Thư mục lưu trữ:    {settings.PERSIST_DIR}")
    print("=" * 70)

    persist_path = Path(settings.PERSIST_DIR)

    if clean and persist_path.exists():
        print(f"[BuildIndex] Xóa sạch dữ liệu cũ tại '{settings.PERSIST_DIR}'...")
        try:
            shutil.rmtree(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            for item in persist_path.glob("*"):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass

    start_time = time.time()

    # 1. Nạp toàn bộ tài liệu đa định dạng (PDF, DOCX)
    loader = UniversalDocumentLoader()
    raw_docs = loader.load_all(base_dir=settings.DATA_DIR, max_files_per_category=max_files)

    if not raw_docs:
        print("[BuildIndex Lỗi] Không tìm thấy tài liệu nào trong thư mục 'ducument'!")
        return

    # 2. Phân đoạn văn bản pháp luật theo cấu trúc Chương / Điều / Khoản
    print("\n[BuildIndex] Đang phân đoạn văn bản pháp luật theo cấu trúc Chương/Điều/Khoản (Legal Article Chunking)...")
    legal_chunker = LegalArticleChunker(max_chunk_size=1400)
    child_docs, parent_store = legal_chunker.split(raw_docs)


    if max_chunks and max_chunks < len(child_docs):
        print(f"[BuildIndex] Giới hạn số chunks theo yêu cầu: {max_chunks}/{len(child_docs)}")
        child_docs = child_docs[:max_chunks]

    # 3. Khởi tạo đối tượng Embedding
    print(f"\n[BuildIndex] Khởi tạo Embeddings ({actual_provider} - {actual_model})...")
    embeddings = get_embeddings(
        model_name=actual_model,
        api_key=actual_key,
        chunk_size=batch_size,
        provider=actual_provider,
    )

    # 4. Lập chỉ mục Vector DB và lưu bền vững vào đĩa
    print(f"\n[BuildIndex] Bắt đầu Embed và lưu trữ {len(child_docs)} chunks vào ChromaDB...")
    vdb = VectorDB(
        documents=child_docs,
        embedding=embeddings,
        collection_name="vietnamese_docs",
        persist_dir=settings.PERSIST_DIR,
    )

    expected = len(child_docs)
    actual = vdb.db._collection.count()
    print(f"[BuildIndex Kiểm Tra] Vector DB hiện có: {actual}/{expected} chunks.")
    if actual < expected:
        raise RuntimeError(
            f"Chỉ mục KHÔNG đầy đủ: chỉ có {actual}/{expected} chunks. "
            f"Vui lòng chạy lại với cờ --clean: python build_index.py --clean"
        )

    # 5. Xây dựng và lưu chỉ mục BM25 (dùng chung collection 'vietnamese_docs', TUYỆT ĐỐI không embed lại)
    print("\n[BuildIndex] Đang tạo và lưu chỉ mục BM25 (Keyword Index & Documents Cache)...")
    hybrid_db = HybridVectorDB(
        documents=child_docs,
        embedding=embeddings,
        collection_name="vietnamese_docs",
        persist_dir=settings.PERSIST_DIR,
    )

    from src.storage.vector_store import check_index_health
    health = check_index_health(settings.PERSIST_DIR)
    print(f"[BuildIndex] Sức khỏe chỉ mục: {health['count']} vectors từ {health['sources']} nguồn tài liệu.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(" ✅ HOÀN TẤT LẬP CHỈ MỤC KHO TRI THỨC THÀNH CÔNG!")

    print(f" - Tổng tài liệu gốc:  {len(raw_docs)} trang/mục")
    print(f" - Số Parent Chunks:   {len(parent_store)}")
    print(f" - Số Chunks Vector:   {len(child_docs)}")
    print(f" - Thời gian thực thi: {elapsed:.2f} giây")
    print(f" - Vị trí lưu trữ:     {persist_path.resolve()}")
    print("=" * 70)
    print("👉 Từ bây giờ, khi chạy Web UI hoặc CLI, hệ thống sẽ nạp trực tiếp trong 0.01 giây,")
    print("   TUYỆT ĐỐI KHÔNG EMBED LẠI TÀI LIỆU NỮA!\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-index all documents in 'ducument' into ChromaDB.")
    parser.add_argument("--clean", action="store_true", help="Xóa sạch index cũ và lập lại từ đầu")
    parser.add_argument("--max-files", type=int, default=None, help="Số file tối đa mỗi thư mục")
    parser.add_argument("--max-chunks", type=int, default=None, help="Số lượng chunk tối đa muốn lập chỉ mục")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API Key")
    parser.add_argument("--provider", type=str, default=None, help="fastembed hoặc openrouter")
    parser.add_argument("--model", type=str, default=None, help="Tên mô hình Embedding")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch khi gọi Embedding API")
    args = parser.parse_args()

    build_full_knowledge_index(
        clean=args.clean,
        max_files=args.max_files,
        max_chunks=args.max_chunks,
        api_key=args.api_key,
        model_name=args.model,
        provider=args.provider,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

