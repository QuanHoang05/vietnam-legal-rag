"""
CHƯƠNG TRÌNH ĐÁNH GIÁ 50 CÂU HỎI BENCHMARK RAG
Sử dụng đúng luồng pipeline thống nhất:
  QueryRouter (Gemini) -> TwoStageHybridRetriever (RRF + Cross-Encoder) -> LLM
Chấm điểm 4 metric thực tế: Faithfulness, Answer Relevancy, Context Precision, Context Recall
Xuất báo cáo CSV + biểu đồ PNG/PDF vào reports/
"""

import sys
import json
import time
import argparse
from typing import Optional
import pandas as pd
from pathlib import Path

# Fix encoding trên Windows (PowerShell mặc định cp1252)
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
from src.storage.vector_store import HybridVectorDB, check_index_health
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import TwoStageHybridRetriever
from src.query_transform.query_router import QueryRouter
from src.pipeline.batch_rag import BatchRAG
from src.evaluation.ragas_evaluator import evaluate_ragas
from src.evaluation.visualizer import plot_single_evaluation


def run_single_question(query: str, router: QueryRouter, retriever: TwoStageHybridRetriever, rag: BatchRAG) -> dict:
    """
    Chạy pipeline thống nhất cho 1 câu hỏi:
    QueryRouter -> (decompose/hyde/rewrite/direct) -> TwoStageHybridRetriever -> LLM
    """
    try:
        route_result = router.process(query)
        strategy = route_result["strategy_used"]

        if strategy == "decompose":
            sub_queries = route_result.get("processed_queries", [query])
            all_docs = []
            seen = set()
            for sq in sub_queries:
                docs = retriever.invoke(sq)
                for d in docs:
                    if d.page_content not in seen:
                        all_docs.append(d)
                        seen.add(d.page_content)
            # Rerank lần cuối theo câu gốc
            reranker = retriever.reranker
            if reranker and all_docs:
                final_docs = reranker.rerank(query, all_docs)[:5]
            else:
                final_docs = all_docs[:5]
            formatted_ctx = rag._format_docs(final_docs)
            prompt_text = rag.prompt.format(history_block="", context=formatted_ctx, question=query)
            answers = rag.batch_generate([prompt_text])
            return {
                "answer": answers[0] if answers else "Không có thông tin",
                "contexts": [d.page_content for d in final_docs],
                "documents": final_docs,
                "strategy": strategy,
            }

        elif strategy == "hyde" and route_result.get("hypothetical_doc"):
            hypo_text = route_result["hypothetical_doc"]
            docs = retriever.invoke(f"{query}\n{hypo_text}")
            formatted_ctx = rag._format_docs(docs)
            prompt_text = rag.prompt.format(history_block="", context=formatted_ctx, question=query)
            answers = rag.batch_generate([prompt_text])
            return {
                "answer": answers[0] if answers else "Không có thông tin",
                "contexts": [d.page_content for d in docs],
                "documents": docs,
                "strategy": strategy,
            }

        elif strategy == "rewrite":
            target_q = route_result["processed_queries"][0]
            results = rag.answer_with_contexts_batch([target_q], retriever)
            r = results[0]
            r["strategy"] = strategy
            return r

        else:
            # direct
            results = rag.answer_with_contexts_batch([query], retriever)
            r = results[0]
            r["strategy"] = strategy
            return r

    except Exception as e:
        print(f"  [LỖI câu hỏi] {e}")
        return {"answer": "Không có thông tin", "contexts": [], "documents": [], "strategy": "error"}


def run_50_questions_evaluation(
    testset_path: str = "data/benchmark_testset_50.json",
    sample_size: Optional[int] = None,
    api_key: Optional[str] = None,
):
    print("=" * 80)
    print(" ĐÁNH GIÁ 50 CÂU HỎI BENCHMARK RAG - PIPELINE THỐNG NHẤT")
    print(f"  Testset:    {testset_path}")
    print(f"  Model LLM:  {settings.OPENROUTER_MODEL}")
    print(f"  Embedding:  {settings.EMBEDDING_MODEL} (local)")
    print(f"  Reranker:   {settings.RERANKER_MODEL}")
    print("=" * 80)

    # 1. Đọc testset
    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if sample_size and sample_size < len(test_data):
        test_data = test_data[:sample_size]

    test_df = pd.DataFrame(test_data)
    questions = test_df["user_input"].tolist()
    print(f"[1/4] Đã nạp {len(questions)} câu hỏi kiểm thử.")

    # 2. Khởi tạo model và chỉ mục
    actual_key = api_key or settings.OPENROUTER_API_KEY
    llm = get_llm(model_name=settings.OPENROUTER_MODEL, api_key=actual_key)
    embeddings = get_embeddings()

    health = check_index_health(settings.PERSIST_DIR, min_expected=100)
    print(f"[2/4] Chỉ mục: {health['count']} vectors từ {health['sources']} nguồn — {'OK' if health['healthy'] else 'CHUA DU'}")

    if not health["healthy"]:
        print(f"\n[LOI] Chi muc chua day du ({health['count']} vectors).")
        print("Vui long chay: python build_index.py --clean\n")
        sys.exit(1)

    hdb = HybridVectorDB(documents=None, embedding=embeddings, persist_dir=settings.PERSIST_DIR)

    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL,
        top_k=5,
        api_key=actual_key,
    )

    two_stage = TwoStageHybridRetriever(
        vector_db=hdb.vector_db,
        bm25=hdb.bm25,
        documents=hdb.documents,
        reranker=reranker,
        candidate_k=30,
        k=8,
    )

    router = QueryRouter(api_key=settings.GEMINI_API_KEY)
    rag = BatchRAG(llm=llm)

    # 3. Chạy suy luận từng câu
    print(f"\n[3/4] Thực thi suy luận RAG trên {len(questions)} câu hỏi...")
    start_time = time.time()
    eval_rows = []

    for i, (question, (_, row)) in enumerate(zip(questions, test_df.iterrows()), 1):
        print(f"  [{i:02d}/{len(questions)}] {question[:70]}...")
        result = run_single_question(question, router, two_stage, rag)
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

    elapsed = time.time() - start_time
    print(f"-> Hoan tat {len(questions)} cau trong {elapsed:.1f}s (trung binh {elapsed/len(questions):.1f}s/cau).")

    eval_df = pd.DataFrame(eval_rows)

    # 4. Chấm điểm
    print(f"\n[4/4] Tinh toan 4 chi so chat luong...")
    eval_result = evaluate_ragas(eval_df)
    scores = eval_result["scores"]
    detailed_df = eval_result["results_df"]

    # 5. Xuất báo cáo
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "eval_50_unified_pipeline.csv"
    png_path = reports_dir / "eval_50_unified_pipeline.png"
    pdf_path = reports_dir / "eval_50_unified_pipeline.pdf"

    detailed_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_single_evaluation(
        scores=scores,
        title="50-Questions Evaluation — Unified Pipeline (QueryRouter + TwoStage)",
        output_pdf=str(pdf_path),
        output_png=str(png_path),
    )

    print("\n" + "=" * 80)
    print(" KET QUA DANH GIA THUC TE TREN BO TESTSET 50 CAU")
    print("=" * 80)
    for k, v in scores.items():
        print(f"  * {k:<22}: {v:.4f}")
    print("=" * 80)
    print(f"-> Chi tiet tung cau (CSV): {csv_path}")
    print(f"-> Bieu do chi so (PNG):    {png_path}")
    print(f"-> Bieu do chi so (PDF):    {pdf_path}")
    print("=" * 80 + "\n")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Danh gia thuc te 50 cau hoi tren kho tai lieu.")
    parser.add_argument("--testset", type=str, default="data/benchmark_testset_50.json",
                        help="Duong dan file testset (mac dinh: data/benchmark_testset_50.json)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="So cau thu nhanh (mac dinh: toan bo 50 cau)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API Key (neu khong co trong .env)")
    args = parser.parse_args()

    run_50_questions_evaluation(
        testset_path=args.testset,
        sample_size=args.sample_size,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
