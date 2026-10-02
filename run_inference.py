"""
GIAI ĐOẠN 1: SUY LUẬN RAG (INFERENCE ONLY) & LƯU CHECKPOINT CACHE
Chạy quy trình RAG: QueryRouter -> TwoStageHybridRetriever -> BatchRAG
Lưu toàn bộ kết quả vào file cache JSON để phục vụ bước đánh giá độc lập.
HOÀN TOÀN KHÔNG CHẤM ĐIỂM TỰ ĐỘNG.
"""

import sys
import json
import time
import argparse
from typing import Optional
from pathlib import Path
import pandas as pd

# Fix encoding trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.settings import settings
from src.models.llm_factory import get_llm
from src.models.embedding_factory import get_embeddings
from src.storage.vector_store import HybridVectorDB
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import TwoStageHybridRetriever
from src.query_transform.query_router import QueryRouter
from src.pipeline.batch_rag import BatchRAG
from src.pipeline.unified_pipeline import run_pipeline_single_query


def run_rag_inference(
    testset_path: str = "data/benchmark_testset_50.json",
    sample_size: Optional[int] = None,
    api_key: Optional[str] = None,
    output_cache: Optional[str] = None,
) -> Path:
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    actual_path = ROOT_DIR / testset_path if not Path(testset_path).is_absolute() else Path(testset_path)
    if not actual_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file testset: {actual_path}")

    with open(actual_path, "r", encoding="utf-8") as f:
        raw_testset = json.load(f)

    test_df = pd.DataFrame(raw_testset)
    if sample_size and sample_size < len(test_df):
        test_df = test_df.iloc[:sample_size]

    n_questions = len(test_df)
    questions = test_df["user_input"].tolist()

    if output_cache:
        cache_path = Path(output_cache)
        if not cache_path.is_absolute():
            cache_path = ROOT_DIR / cache_path
    else:
        cache_path = reports_dir / f"eval_{n_questions}_inference_cache.json"

    print("\n" + "=" * 80)
    print(f" 🚀 BẮT ĐẦU GIAI ĐOẠN 1: SUY LUẬN RAG (INFERENCE ONLY)")
    print(f"  Số câu hỏi:       {n_questions}")
    print(f"  File testset:     {actual_path.name}")
    print(f"  Nơi lưu cache:    {cache_path}")
    print("=" * 80 + "\n")

    print("[1/2] Đang khởi tạo các thành phần Unified Pipeline...")
    llm = get_llm(api_key=api_key)
    embeddings = get_embeddings()
    vector_db = HybridVectorDB(persist_dir=str(ROOT_DIR / settings.PERSIST_DIR), embedding_function=embeddings)
    reranker = CrossEncoderReranker(device="cpu")
    two_stage = TwoStageHybridRetriever(vector_db=vector_db, reranker=reranker)
    router = QueryRouter(llm=llm)
    rag = BatchRAG(llm=llm, retriever=two_stage)

    print(f"\n[2/2] Đang thực thi suy luận trên {n_questions} câu...")
    start_time = time.time()
    eval_rows = []

    for i, (question, (_, row)) in enumerate(zip(questions, test_df.iterrows()), 1):
        q_preview = question[:65] + "..." if len(question) > 65 else question
        print(f"  [{i:02d}/{n_questions}] {q_preview}")
        result = run_pipeline_single_query(question, router, two_stage, rag)

        eval_rows.append({
            "id": row.get("id"),
            "category": row.get("category", "General"),
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": result["contexts"],
            "reference": row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts", []),
            "strategy": result.get("strategy", "direct"),
        })

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(eval_rows, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    [Cảnh báo] Lỗi khi ghi cache câu {i}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" ✅ HOÀN THÀNH SUY LUẬN {n_questions} CÂU TRONG {elapsed:.1f}s (Trung bình: {elapsed/n_questions:.1f}s/câu)")
    print(f" 📂 File cache đã sẵn sàng: {cache_path}")
    print("=" * 80)
    print("👉 Bây giờ bạn có thể CHẤM ĐIỂM từ cache bằng lệnh:")
    print(f"   python run_evaluation.py --cache {cache_path.name} --method llm_judge")
    print(f"   hoặc: python run_evaluation.py --cache {cache_path.name} --method ragas")
    print("=" * 80 + "\n")

    return cache_path


def main():
    parser = argparse.ArgumentParser(description="Giai đoạn 1: Chạy suy luận hệ thống RAG và lưu cache.")
    parser.add_argument("--testset", type=str, default="data/benchmark_testset_50.json",
                        help="Đường dẫn file testset (mặc định: data/benchmark_testset_50.json)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Số lượng câu cần suy luận (ví dụ: 5, 20 hoặc 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Tên file cache xuất ra (mặc định: reports/eval_{N}_inference_cache.json)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key tùy chọn")
    args = parser.parse_args()

    run_rag_inference(
        testset_path=args.testset,
        sample_size=args.sample_size,
        api_key=args.api_key,
        output_cache=args.output,
    )


if __name__ == "__main__":
    main()
