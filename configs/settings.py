"""
Cấu hình tập trung cho toàn bộ dự án Insight into RAG.
Hỗ trợ đọc từ biến môi trường (.env) và định nghĩa các siêu tham số
được chuẩn hóa theo đúng tài liệu 'Project: Insight into RAG'.
"""

import os
import random
from typing import Optional
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Tải cấu hình từ .env nếu tồn tại (ghi đè biến môi trường cũ)
env_file_path = BASE_DIR / ".env"
if env_file_path.exists():
    load_dotenv(dotenv_path=env_file_path, override=True)
else:
    load_dotenv(override=True)



class Settings:
    # -------------------------------------------------------------
    # 1. OpenRouter & LLM Configurations
    # -------------------------------------------------------------
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Người dùng có thể đổi model linh hoạt: gpt-4o-mini, claude-3-5-sonnet, deepseek-chat, llama-3.3-70b-instruct...
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))

    # Headers cho OpenRouter
    OPENROUTER_REFERER: str = os.getenv("OPENROUTER_REFERER", "https://github.com/aivietnam-ai/rag-insight")
    OPENROUTER_TITLE: str = os.getenv("OPENROUTER_TITLE", "Insight into RAG Project")

    # Google AI Studio / Gemini API Configurations (Dành cho Query Router & Rewriter)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # -------------------------------------------------------------
    # 2. Embedding Configurations (OpenRouter API or FastEmbed)
    # -------------------------------------------------------------
    # Hỗ trợ 'openrouter' hoặc 'fastembed' (local ONNX, 0đ, 0 rate limit)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    # Mặc định sử dụng BAAI/bge-m3 (hỗ trợ tiếng Việt & đa ngữ) hoặc nvidia/llama-nemotron-embed-vl-1b-v2:free
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # -------------------------------------------------------------
    # 3. Baseline & System Parameters
    # -------------------------------------------------------------
    SEED: int = int(os.getenv("SEED", "42"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))

    _raw_data_dir = os.getenv("DATA_DIR", "ducument")
    DATA_DIR: str = str((BASE_DIR / _raw_data_dir).resolve()) if not Path(_raw_data_dir).is_absolute() else _raw_data_dir

    _raw_persist_dir = os.getenv("PERSIST_DIR", "chroma_data")
    PERSIST_DIR: str = str((BASE_DIR / _raw_persist_dir).resolve()) if not Path(_raw_persist_dir).is_absolute() else _raw_persist_dir

    _raw_testset_path = os.getenv("TESTSET_PATH", "data/benchmark_testset_50.json")
    TESTSET_PATH: str = str((BASE_DIR / _raw_testset_path).resolve()) if not Path(_raw_testset_path).is_absolute() else _raw_testset_path

    # Fixed Chunking Parameters (Baseline)
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    BASELINE_RETRIEVE_K: int = 3

    # -------------------------------------------------------------
    # 4. Experiment 1: Semantic Chunking Parameters
    # -------------------------------------------------------------
    SEMANTIC_BREAKPOINT_THRESHOLD: float = 0.5
    MIN_CHUNK_SIZE: int = 600
    MAX_CHUNK_SIZE: int = 1024
    SEMANTIC_CHUNK_OVERLAP: int = 128

    # -------------------------------------------------------------
    # 5. Experiment 2: Retrieval Parameters
    # -------------------------------------------------------------
    BM25_K: int = 5
    HYBRID_INTERLEAVING_TOP_K: int = 5
    HYBRID_K: int = 7
    RRF_K: int = 40  # Hằng số làm mượt mẫu số trong công thức RRF

    # -------------------------------------------------------------
    # 6. Experiment 3: Query Transformation Parameters
    # -------------------------------------------------------------
    HYDE_K: int = 3
    DECOMPOSITION_MAX_SUB_QUESTIONS: int = 3

    # -------------------------------------------------------------
    # 7. Experiment 4: Re-ranking Parameters (OpenRouter API)
    # -------------------------------------------------------------
    # Mặc định sử dụng NVIDIA Llama Nemotron Rerank VL 1B V2 (10k context, miễn phí)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "nvidia/llama-nemotron-rerank-vl-1b-v2:free")
    RETRIEVE_K: int = 20  # Số docs lấy ra ban đầu trước khi rerank
    RERANK_TOP_K: int = 7  # Số docs giữ lại sau rerank
    
    # MMR Parameters
    MMR_K: int = 3
    MMR_FETCH_K: int = 10
    MMR_LAMBDA_MULT: float = 0.5

    # -------------------------------------------------------------
    # 8. Evaluation Parameters (Ragas)
    # -------------------------------------------------------------
    EVAL_SAMPLE_SIZE: Optional[int] = None
    # Giữ max_workers thấp (2-4) để tránh Rate Limit của OpenRouter → NaN hàng loạt
    EVAL_MAX_WORKERS: int = int(os.getenv("EVAL_MAX_WORKERS", "2"))
    EVAL_TIMEOUT: int = 180

    @classmethod
    def set_global_seed(cls, seed_value: int = None):
        """Cố định seed ngẫu nhiên cho toàn bộ quy trình thực nghiệm"""
        target_seed = seed_value or cls.SEED
        random.seed(target_seed)
        np.random.seed(target_seed)
        os.environ["PYTHONHASHSEED"] = str(target_seed)
        print(f"[Settings] Global seed set to {target_seed}")


settings = Settings()
