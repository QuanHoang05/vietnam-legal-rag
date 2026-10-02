"""
TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY WRAPPER)
Cho phép chạy lệnh đánh giá cũ 'python run_evaluation_50.py'
Tự động chuyển tiếp mượt mà sang 2 giai đoạn chuẩn:
  - Giai đoạn 1: run_inference.py (nếu chưa có cache)
  - Giai đoạn 2: run_evaluation.py (chấm điểm bằng LLM Judge hoặc Ragas)
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_inference import run_rag_inference
from run_evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Bộ điều phối đánh giá RAG Benchmark (Unified Wrapper).")
    parser.add_argument("--step", type=str, choices=["all", "inference", "eval"], default="all",
                        help="Bước thực thi: 'inference', 'eval', hoặc 'all'. Mặc định: all")
    parser.add_argument("--method", type=str, choices=["llm_judge", "ragas"], default="llm_judge",
                        help="Phương pháp chấm điểm: 'llm_judge' (khuyên dùng) hoặc 'ragas'. Mặc định: llm_judge")
    parser.add_argument("--cache", "--rescore", dest="cache_path", type=str, default=None,
                        help="Đường dẫn file cache JSON đã suy luận")
    parser.add_argument("--testset", type=str, default="data/benchmark_testset_50.json",
                        help="Đường dẫn file testset")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Số câu thử nhanh (ví dụ: 5 hoặc 20)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key tùy chọn")
    args = parser.parse_args()

    cache_file = args.cache_path

    # Nếu người dùng truyền file cache -> Chuyển thẳng sang chấm điểm
    if cache_file:
        run_evaluation(cache_path=cache_file, method=args.method)
        return

    # Nếu chạy riêng inference hoặc chạy all mà chưa có cache
    if args.step in ["all", "inference"]:
        cache_file = run_rag_inference(
            testset_path=args.testset,
            sample_size=args.sample_size,
            api_key=args.api_key,
        )

    # Chạy tiếp bước chấm điểm nếu yêu cầu
    if args.step in ["all", "eval"] and cache_file:
        run_evaluation(cache_path=str(cache_file), method=args.method)


if __name__ == "__main__":
    main()
