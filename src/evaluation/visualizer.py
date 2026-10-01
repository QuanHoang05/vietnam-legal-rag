"""
Trực quan hóa kết quả đánh giá hệ thống RAG:
1. Biểu đồ cột đơn cho từng thực nghiệm (Lưu dạng PDF và PNG).
2. Biểu đồ so sánh độ chênh lệch (Delta) so với Baseline cho toàn bộ 9 cấu hình (Hình 19 trong tài liệu).
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def plot_single_evaluation(
    scores: Dict[str, float],
    title: str = "RAGAS Evaluation Scores",
    output_pdf: str = "evaluation_report.pdf",
    output_png: str = "evaluation_report.png",
):
    """
    Vẽ biểu đồ cột 4 chỉ số Ragas cho một thực nghiệm đơn lẻ.
    (Theo mục VII.4.7 trong tài liệu)
    """
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    labels = [metric_labels.get(k, k) for k in scores.keys()]
    values = list(scores.values())
    colors = ["#10b981", "#0ea5e9", "#6366f1", "#f59e0b"][:len(labels)]

    plt.figure(figsize=(9, 6.5))
    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Đường chuẩn tham chiếu ngưỡng chất lượng tối thiểu 0.5
    plt.axhline(0.5, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.7, label="Ngưỡng đạt tối thiểu (0.50)", zorder=2)

    bars = plt.bar(labels, values, color=colors, edgecolor="#1e293b", alpha=0.9, width=0.5, zorder=3)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Điểm số chất lượng (0.0 - 1.0)", fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", fontsize=10)

    # Hiển thị giá trị cụ thể trên từng cột
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    # Lưu PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
    # Lưu PNG
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu báo cáo trực quan tại: {output_pdf} và {output_png}")



def plot_benchmark_comparison(
    benchmark_df: pd.DataFrame,
    output_png: str = "benchmark_comparison.png",
    output_pdf: str = "benchmark_comparison.pdf",
):
    """
    Vẽ biểu đồ so sánh toàn diện 4 chỉ số giữa Baseline và các thực nghiệm nâng cao
    (Tái hiện Hình 19 trong tài liệu Insight into RAG).
    """
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_titles = [
        "1. Faithfulness (Baseline = 0.73)",
        "2. Answer Relevancy (Baseline = 0.53)",
        "3. Context Precision (Baseline = 0.81)",
        "4. Context Recall (Baseline = 0.67)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric not in benchmark_df.columns:
            continue

        sorted_df = benchmark_df.sort_values(by=metric, ascending=True)
        y_pos = range(len(sorted_df))

        bars = ax.barh(
            y_pos,
            sorted_df[metric],
            color="#0ea5e9",
            edgecolor="#0369a1",
            alpha=0.85,
            height=0.6,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df["method"], fontsize=10)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_title(metric_titles[idx], fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Ghi giá trị score
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle("Tổng hợp so sánh hiệu năng các chiến lược nâng cao so với Baseline (Insight into RAG)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Visualizer] Đã lưu biểu đồ tổng kết benchmark tại: {output_pdf} và {output_png}")
