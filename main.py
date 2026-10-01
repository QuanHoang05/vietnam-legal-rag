"""
GIAO DIỆN CHÍNH (CLI) - PROJECT: INSIGHT INTO RAG
Hỏi đáp thông minh qua pipeline thống nhất: QueryRouter -> TwoStageHybridRetriever -> LLM
"""

import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from configs.settings import settings


def build_pipeline(model_name: str = None):
    """Xây dựng pipeline thống nhất: HybridVectorDB + TwoStageHybridRetriever + QueryRouter"""
    actual_model = model_name or settings.OPENROUTER_MODEL

    print(f"\n[Khởi tạo Pipeline]")
    print(f"  - Model LLM:    {actual_model}")
    print(f"  - Embedding:    BAAI/bge-m3 (local)")
    print(f"  - Reranker:     {settings.RERANKER_MODEL}")

    try:
        from src.models.llm_factory import get_llm
        from src.models.embedding_factory import get_embeddings
        from src.storage.vector_store import HybridVectorDB, check_index_health
        from src.retrieval.reranker import CrossEncoderReranker
        from src.retrieval.hybrid_retriever import TwoStageHybridRetriever
        from src.query_transform.query_router import QueryRouter
        from src.pipeline.batch_rag import BatchRAG
    except ImportError as e:
        print(f"\n[Lỗi thiếu thư viện] {e}")
        print("Vui lòng chạy: pip install -r requirements.txt\n")
        sys.exit(1)

    settings.set_global_seed()
    llm = get_llm(model_name=actual_model)
    embeddings = get_embeddings()

    # Kiểm tra chỉ mục trên đĩa
    health = check_index_health(settings.PERSIST_DIR, min_expected=100)
    if health["healthy"]:
        print(f"[Storage] Chỉ mục hợp lệ: {health['count']} vectors từ {health['sources']} nguồn.")
    else:
        print(f"[Storage] CẢNH BÁO: Chỉ mục chưa đủ ({health.get('count', 0)} vectors).")
        print("  Hãy chạy: python build_index.py --clean")
        sys.exit(1)

    hdb = HybridVectorDB(documents=None, embedding=embeddings, persist_dir=settings.PERSIST_DIR)

    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL,
        top_k=5,
        api_key=settings.OPENROUTER_API_KEY,
    )

    retriever = TwoStageHybridRetriever(
        vector_db=hdb.vector_db,
        bm25=hdb.bm25,
        documents=hdb.documents,
        reranker=reranker,
        candidate_k=15,
        k=5,
    )

    router = QueryRouter(api_key=settings.GEMINI_API_KEY)
    rag = BatchRAG(llm=llm)

    return rag, retriever, router


def run_query(query: str, rag, retriever, router) -> tuple:
    """Chạy 1 câu hỏi qua pipeline thống nhất. Trả về (answer, contexts)."""
    from src.retrieval.reranker import CrossEncoderReranker

    route_result = router.process(query)
    strategy = route_result["strategy_used"]
    print(f"  [Router] Chiến lược: {strategy.upper()} — {route_result.get('explanation', '')}")

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
        if retriever.reranker and all_docs:
            final_docs = retriever.reranker.rerank(query, all_docs)[:5]
        else:
            final_docs = all_docs[:5]
        formatted_ctx = rag._format_docs(final_docs)
        prompt_text = rag.prompt.format(history_block="", context=formatted_ctx, question=query)
        answers = rag.batch_generate([prompt_text])
        return answers[0] if answers else "Không có thông tin", [d.page_content for d in final_docs]

    elif strategy == "hyde" and route_result.get("hypothetical_doc"):
        hypo = route_result["hypothetical_doc"]
        docs = retriever.invoke(f"{query}\n{hypo}")
        formatted_ctx = rag._format_docs(docs)
        prompt_text = rag.prompt.format(history_block="", context=formatted_ctx, question=query)
        answers = rag.batch_generate([prompt_text])
        return answers[0] if answers else "Không có thông tin", [d.page_content for d in docs]

    elif strategy == "rewrite":
        target_q = route_result["processed_queries"][0]
        results = rag.answer_with_contexts_batch([target_q], retriever)
        return results[0]["answer"], results[0].get("contexts", [])

    else:
        results = rag.answer_with_contexts_batch([query], retriever)
        return results[0]["answer"], results[0].get("contexts", [])


def interactive_chat(model_name: str = None):
    rag, retriever, router = build_pipeline(model_name=model_name)

    print("\n" + "=" * 70)
    print(" SẴN SÀNG NHẬN CÂU HỎI TRỰC TIẾP!")
    print(" (Gõ 'exit' hoặc 'quit' để thoát)")
    print("=" * 70)

    while True:
        try:
            query = input("\n[Bạn hỏi]: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Tạm biệt!")
                break

            print("[AI đang tra cứu tài liệu và suy luận...]")
            answer, contexts = run_query(query, rag, retriever, router)

            print(f"\n[AI trả lời]:\n{answer}\n")
            print(f"[Ngữ cảnh trích xuất ({len(contexts)} đoạn)]:")
            for idx, c in enumerate(contexts[:3], 1):
                snippet = c.strip().replace("\n", " ")[:150]
                print(f"  {idx}. {snippet}...")

        except KeyboardInterrupt:
            print("\nĐã hủy.")
            break
        except Exception as e:
            print(f"[Lỗi]: {e}")


def main():
    parser = argparse.ArgumentParser(description="Insight into RAG - CLI Hỏi Đáp")
    parser.add_argument("--model", type=str, default=None,
                        help="Tên mô hình OpenRouter (VD: openai/gpt-4o-mini)")
    parser.add_argument("--query", type=str, default=None,
                        help="Câu hỏi tra cứu trực tiếp (single question mode)")
    args = parser.parse_args()

    if args.query:
        rag, retriever, router = build_pipeline(model_name=args.model)
        print(f"\n[Bạn hỏi]: {args.query}")
        print("[AI đang tra cứu tài liệu và suy luận...]")
        answer, contexts = run_query(args.query, rag, retriever, router)
        print(f"\n[AI trả lời]:\n{answer}\n")
        print(f"[Ngữ cảnh trích xuất ({len(contexts)} đoạn)]:")
        for i, ctx in enumerate(contexts[:3], 1):
            print(f"  {i}. {ctx[:150].strip()}...")
    else:
        interactive_chat(model_name=args.model)


if __name__ == "__main__":
    main()
