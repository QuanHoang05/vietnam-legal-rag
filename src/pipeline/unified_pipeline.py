"""
Module điều phối suy luận RAG thống nhất:
QueryRouter (Gemini) -> TwoStageHybridRetriever (BM25 + Dense + Reranker) -> BatchRAG (LLM)
Tập trung hóa logic để tránh duplicate code giữa các script chạy.
"""

from typing import Dict, Any, List
from configs.settings import settings
from src.retrieval.hybrid_retriever import TwoStageHybridRetriever
from src.query_transform.query_router import QueryRouter
from src.pipeline.batch_rag import BatchRAG


def run_pipeline_single_query(
    query: str,
    router: QueryRouter,
    retriever: TwoStageHybridRetriever,
    rag: BatchRAG,
) -> Dict[str, Any]:
    """
    Chạy pipeline suy luận thống nhất cho 1 câu hỏi:
    1. QueryRouter phân loại chiến lược (decompose / hyde / rewrite / direct)
    2. TwoStageHybridRetriever tìm kiếm và rerank văn bản
    3. BatchRAG sinh câu trả lời có căn cứ trích dẫn
    """
    try:
        route_result = router.process(query)
        strategy = route_result.get("strategy_used", "direct")

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
            reranker = retriever.reranker
            if reranker and all_docs:
                final_docs = reranker.rerank(query, all_docs)[:settings.RERANK_TOP_K]
            else:
                final_docs = all_docs[:settings.RERANK_TOP_K]
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
            target_q = route_result.get("processed_queries", [query])[0]
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
        print(f"    [Fallback] Lỗi Pipeline ({e}), chuyển sang truy xuất cơ bản...")
        try:
            docs = retriever.invoke(query)
            formatted_ctx = rag._format_docs(docs)
            prompt_text = rag.prompt.format(history_block="", context=formatted_ctx, question=query)
            answers = rag.batch_generate([prompt_text])
            return {
                "answer": answers[0] if answers else "Không có thông tin",
                "contexts": [d.page_content for d in docs],
                "documents": docs,
                "strategy": "direct_fallback",
            }
        except Exception:
            return {"answer": "Không có thông tin", "contexts": [], "documents": [], "strategy": "error"}
