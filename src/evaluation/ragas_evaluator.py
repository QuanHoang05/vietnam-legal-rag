"""
Đánh giá hệ thống RAG bằng Ragas Framework chính thức (v0.2.x/0.4.x).
Dùng 4 metric chuẩn của Ragas — tất cả đều do LLM chấm điểm.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from configs.settings import settings


def evaluate_ragas(
    eval_df: pd.DataFrame,
    sample_size: Optional[int] = settings.EVAL_SAMPLE_SIZE,
    max_workers: int = settings.EVAL_MAX_WORKERS,
    timeout: int = settings.EVAL_TIMEOUT,
    force_ragas: bool = False,
) -> Dict[str, Any]:
    """
    Chấm điểm bộ kết quả RAG bằng Ragas Framework chính thức.
    """
    if sample_size is not None and sample_size < len(eval_df):
        eval_subset = eval_df.sample(n=sample_size, random_state=settings.SEED).reset_index(drop=True)
    else:
        eval_subset = eval_df.reset_index(drop=True)

    has_valid_key = bool(settings.OPENROUTER_API_KEY and "your" not in settings.OPENROUTER_API_KEY)
    if not has_valid_key:
        raise ValueError(
            "Cần OPENROUTER_API_KEY hợp lệ để chạy Ragas LLM evaluation. "
            "Vui lòng cấu hình trong file .env"
        )

    print(f"[RagasEvaluator] Khởi động Ragas chính thức trên {len(eval_subset)} mẫu...")

    # ============================================================
    # Import Ragas API và Monkeypatch
    # ============================================================
    try:
        import sys
        import langchain_community
        if "langchain_community.chat_models" not in sys.modules:
            sys.modules["langchain_community.chat_models"] = type("module", (object,), {})
        if "langchain_community.chat_models.vertexai" not in sys.modules:
            sys.modules["langchain_community.chat_models.vertexai"] = type("module", (object,), {"ChatVertexAI": None})
            
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import RunConfig
    except ImportError as e:
        raise RuntimeError(
            f"Ragas chưa cài đúng: {e}. Vui lòng chạy: pip install ragas datasets"
        ) from e

    from src.models.llm_factory import get_openrouter_llm
    from src.models.embedding_factory import get_embeddings

    # LLM và Embedding dùng để Ragas tự chấm điểm
    evaluator_llm = LangchainLLMWrapper(get_openrouter_llm())
    evaluator_emb = LangchainEmbeddingsWrapper(get_embeddings())

    # Khởi tạo 4 metric
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # ============================================================
    # Xây dựng EvaluationDataset
    # ============================================================
    samples = []
    for _, row in eval_subset.iterrows():
        ref_ctx = row.get("reference_contexts", [])
        if isinstance(ref_ctx, str):
            ref_ctx = [ref_ctx]

        retrieved = row.get("retrieved_contexts", [])
        if isinstance(retrieved, str):
            retrieved = [retrieved]

        samples.append(SingleTurnSample(
            user_input=str(row.get("user_input", "")),
            response=str(row.get("response", "")),
            retrieved_contexts=[str(c) for c in retrieved],
            reference=str(row.get("reference", "")),
            reference_contexts=[str(c) for c in ref_ctx] if ref_ctx else None,
        ))

    eval_dataset = EvaluationDataset(samples=samples)

    run_cfg = RunConfig(
        max_workers=max_workers,
        timeout=timeout,
        max_retries=2,
        max_wait=60,
    )

    # ============================================================
    # Chạy đánh giá bằng LLM
    # ============================================================
    print("[RagasEvaluator] Đang chấm điểm bằng LLM (OpenRouter)... Chờ vài phút...")
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_emb,
            run_config=run_cfg,
            raise_exceptions=False,
        )
    except Exception as e:
        raise RuntimeError(f"[RagasEvaluator] Lỗi khi chạy evaluate(): {e}") from e

    results_df = result.to_pandas()

    # Trích xuất điểm số trung bình cho từng metric
    METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {}
    n_scored = {}
    for col in METRIC_COLS:
        if col in results_df.columns:
            valid = results_df[col].dropna()
            n_scored[col] = int(valid.count())
            scores[col] = float(valid.mean()) if len(valid) > 0 else float("nan")

    # Gộp metadata gốc vào kết quả
    for col in ["id", "category", "user_input", "response", "reference", "strategy"]:
        if col in eval_subset.columns and col not in results_df.columns:
            results_df[col] = eval_subset[col].values

    # Thêm cột n_scored vào DataFrame để xuất CSV
    total = len(eval_subset)
    for col in METRIC_COLS:
        results_df[f"n_scored_{col}"] = n_scored.get(col, 0)

    # ⚠️ Cảnh báo nếu tỉ lệ mẫu được chấm quá thấp (do Rate Limit gây NaN)
    min_scored = min(n_scored.values()) if n_scored else 0
    coverage_rate = min_scored / total if total > 0 else 0
    if coverage_rate < 0.80:
        print(f"\n⚠️  CẢNH BÁO: Chỉ có {min_scored}/{total} mẫu được chấm điểm ({coverage_rate:.0%})!")
        print("   → Nguyên nhân: Rate Limit OpenRouter gây NaN. Giải pháp:")
        print("   → Đặt EVAL_MAX_WORKERS=1 hoặc 2 trong .env để giảm tốc độ gọi API\n")
        if coverage_rate < 0.50:
            raise RuntimeError(
                f"Quá ít mẫu được chấm ({min_scored}/{total}). Kết quả không đáng tin cậy. "
                "Giảm EVAL_MAX_WORKERS và chạy lại."
            )

    # Cảnh báo nếu tỉ lệ từ chối cao
    refusal_count = eval_subset["response"].apply(
        lambda r: "không có thông tin" in str(r).lower() or not str(r).strip()
    ).sum()
    refusal_rate = refusal_count / total if total > 0 else 0
    if refusal_rate > 0.35:
        print(f"\n⚠️  CẢNH BÁO ĐỎ: {refusal_rate:.1%} câu trả lời là 'Không có thông tin' ({refusal_count}/{total})!")
        print("   → Chỉ mục vector chưa đủ tài liệu. Hãy chạy: python build_index.py --clean\n")

    print("\n=== KẾT QUẢ ĐÁNH GIÁ RAGAS (LLM-based — Official Framework) ===")
    print(f"  Mẫu chấm được: {min_scored}/{total} ({coverage_rate:.0%})")
    for k, v in scores.items():
        print(f"  - {k:<22}: {v:.4f}  [n={n_scored.get(k,0)}]")

    return {
        "scores": scores,
        "n_scored": n_scored,
        "results_df": results_df,
        "engine": "ragas_official_v2",
    }

