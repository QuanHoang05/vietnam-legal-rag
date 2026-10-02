"""
GIAI ĐOẠN 2: ĐÁNH GIÁ CHẤT LƯỢNG RAG (EVALUATION FROM CACHE)
Đọc trực tiếp dữ liệu suy luận từ file cache JSON đã tạo ở Giai đoạn 1.
Hỗ trợ 2 phương pháp đánh giá song song:
  1. Single-Pass LLM-as-a-Judge (Mặc định: --method llm_judge): Siêu tốc, không lo Rate Limit.
  2. Ragas Framework (--method ragas): Đánh giá atomic theo chuẩn thư viện Ragas.
Xuất đầy đủ báo cáo CSV, biểu đồ PNG và PDF rõ ràng theo tên phương pháp.
"""

import sys
import json
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

from src.evaluation.visualizer import plot_single_evaluation


def run_evaluation(
    cache_path: str,
    method: str = "llm_judge",
) -> dict:
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    c_path = Path(cache_path)
    if not c_path.is_absolute():
        c_path = ROOT_DIR / c_path

    if not c_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file cache: {c_path}.\n"
            f"Vui lòng chạy Giai đoạn 1 trước: python run_inference.py --sample-size 20"
        )

    with open(c_path, "r", encoding="utf-8") as f:
        eval_rows = json.load(f)

    eval_df = pd.DataFrame(eval_rows)
    n_questions = len(eval_df)

    print("\n" + "=" * 80)
    print(f" 🚀 BẮT ĐẦU GIAI ĐOẠN 2: ĐÁNH GIÁ CHẤT LƯỢNG RAG ({n_questions} CÂU)")
    print(f"  Phương pháp:     {'Single-Pass LLM-as-a-Judge' if method == 'llm_judge' else 'Ragas Framework'}")
    print(f"  Nguồn dữ liệu:   {c_path.name}")
    print("=" * 80 + "\n")

    if method == "ragas":
        from src.evaluation.ragas_evaluator import evaluate_ragas
        print("[Ragas] Đang tính toán 4 chỉ số chất lượng qua Ragas Evaluator...")
        eval_result = evaluate_ragas(eval_df, max_workers=1)
        scores = eval_result["scores"]
        detailed_df = eval_result["results_df"]
        n_scored = eval_result.get("n_scored", {})

        # Kiểm tra tính hợp lệ: Không gán điểm ảo, mặc định 0.0 nếu lỗi
        for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            valid_count = n_scored.get(k, 0)
            if k not in scores or pd.isna(scores[k]) or valid_count == 0:
                scores[k] = 0.0

        prefix = f"eval_{n_questions}_ragas"
        chart_title = f"{n_questions}-Questions Evaluation — Ragas Framework"

    else:
        from src.evaluation.llm_judge_evaluator import evaluate_llm_judge
        print("[LLM-Judge] Đang thẩm định 4 chỉ số chất lượng qua Single-Pass LLM Judge...")
        eval_result = evaluate_llm_judge(eval_df)
        scores = eval_result["scores"]
        detailed_df = eval_result["results_df"]

        # Kiểm tra tính hợp lệ: Mặc định 0.0 nếu lỗi
        for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if k not in scores or pd.isna(scores[k]):
                scores[k] = 0.0

        prefix = f"eval_{n_questions}_llm_judge"
        chart_title = f"{n_questions}-Questions Evaluation — Single-Pass LLM Judge"

    # Xuất báo cáo CSV & Biểu đồ
    csv_path = reports_dir / f"{prefix}.csv"
    png_path = reports_dir / f"{prefix}.png"
    pdf_path = reports_dir / f"{prefix}.pdf"

    detailed_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_single_evaluation(
        scores=scores,
        title=chart_title,
        output_pdf=str(pdf_path),
        output_png=str(png_path),
    )

    print("\n" + "=" * 80)
    print(f" KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG ({n_questions} CÂU)")
    print("=" * 80)
    for k, v in scores.items():
        print(f"  * {k:<22}: {v:.4f}")
    print("=" * 80)
    print(f"-> Chi tiết từng câu (CSV): {csv_path}")
    print(f"-> Biểu đồ chỉ số (PNG):    {png_path}")
    print(f"-> Biểu đồ chỉ số (PDF):    {pdf_path}")
    print("=" * 80 + "\n")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Giai đoạn 2: Đánh giá chất lượng RAG từ file cache.")
    parser.add_argument("--cache", "--rescore", dest="cache_path", type=str, default=None,
                        help="Đường dẫn file cache JSON từ Giai đoạn 1 (ví dụ: reports/eval_20_inference_cache.json)")
    parser.add_argument("--method", type=str, choices=["llm_judge", "ragas"], default="llm_judge",
                        help="Phương pháp chấm điểm: 'llm_judge' (Mặc định: nhanh gấp 4 lần) hoặc 'ragas'")
    args = parser.parse_args()

    # Tự động tìm file cache mới nhất nếu không chỉ định
    target_cache = args.cache_path
    if not target_cache:
        reports_dir = ROOT_DIR / "reports"
        candidates = sorted(reports_dir.glob("eval_*_inference_cache.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            target_cache = str(candidates[0])
            print(f"[Auto-detect] Tự động chọn file cache gần nhất: {candidates[0].name}")
        else:
            print("❌ LỖI: Chưa có file cache suy luận nào trong thư mục reports/.")
            print("👉 Vui lòng chạy Giai đoạn 1 trước:")
            print("   python run_inference.py --sample-size 20")
            sys.exit(1)

    run_evaluation(
        cache_path=target_cache,
        method=args.method,
    )


if __name__ == "__main__":
    main()
