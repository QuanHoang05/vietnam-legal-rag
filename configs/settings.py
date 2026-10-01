"""
Cấu hình tập trung cho toàn bộ dự án Insight into RAG.
Mọi tham số hệ thống (API keys, URLs, model LLM, embedding, reranker, 
siêu tham số tìm kiếm, chunking, server, đánh giá) đều được quản lý tại đây 
và có thể ghi đè linh hoạt qua file .env.
"""

import os
import random
from typing import Optional, List
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
    # 1. OpenRouter & LLM Gateway Configurations
    # -------------------------------------------------------------
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Mô hình tạo câu trả lời mặc định: openrouter/free, openai/gpt-4o-mini, deepseek/deepseek-chat, v.v.
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    OPENROUTER_FALLBACK_MODEL: str = os.getenv("OPENROUTER_FALLBACK_MODEL", "openai/gpt-4o-mini")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "45"))

    # Headers định danh cho OpenRouter
    OPENROUTER_REFERER: str = os.getenv("OPENROUTER_REFERER", "https://github.com/aivietnam-ai/rag-insight")
    OPENROUTER_TITLE: str = os.getenv("OPENROUTER_TITLE", "Insight into RAG Project")

    # -------------------------------------------------------------
    # 2. Google AI Studio / Gemini API Configurations
    # -------------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_REST_URL: str = os.getenv("GEMINI_REST_URL", "https://generativelanguage.googleapis.com/v1beta")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Danh sách model Gemini dự phòng khi model chính chạm giới hạn 429
    _raw_gemini_fallbacks: str = os.getenv(
        "GEMINI_FALLBACK_MODELS", 
        "gemini-flash-lite-latest,gemini-flash-latest,gemini-2.5-flash-lite"
    )
    GEMINI_FALLBACK_MODELS: List[str] = [m.strip() for m in _raw_gemini_fallbacks.split(",") if m.strip()]
    ROUTER_TIMEOUT: int = int(os.getenv("ROUTER_TIMEOUT", "15"))

    # -------------------------------------------------------------
    # 3. Embedding & Vector Storage Configurations
    # -------------------------------------------------------------
    # Mặc định sử dụng BAAI/bge-m3 (hỗ trợ tiếng Việt & đa ngữ, chạy cục bộ)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "vietnamese_docs")

    _raw_data_dir = os.getenv("DATA_DIR", "ducument")
    DATA_DIR: str = str((BASE_DIR / _raw_data_dir).resolve()) if not Path(_raw_data_dir).is_absolute() else _raw_data_dir

    _raw_persist_dir = os.getenv("PERSIST_DIR", "chroma_data")
    PERSIST_DIR: str = str((BASE_DIR / _raw_persist_dir).resolve()) if not Path(_raw_persist_dir).is_absolute() else _raw_persist_dir

    _raw_testset_path = os.getenv("TESTSET_PATH", "data/benchmark_testset_50.json")
    TESTSET_PATH: str = str((BASE_DIR / _raw_testset_path).resolve()) if not Path(_raw_testset_path).is_absolute() else _raw_testset_path

    # -------------------------------------------------------------
    # 4. Retrieval & Reranker Hyperparameters
    # -------------------------------------------------------------
    BASELINE_RETRIEVE_K: int = int(os.getenv("BASELINE_RETRIEVE_K", "5"))
    CANDIDATE_K: int = int(os.getenv("CANDIDATE_K", "15"))  # Số lượng ứng viên lọc thô qua RRF
    RRF_K: int = int(os.getenv("RRF_K", "40"))  # Hằng số làm mượt mẫu số trong công thức RRF

    # Re-ranking: Mặc định sử dụng AITeamVN/Vietnamese_Reranker (chạy trực tiếp cục bộ, không tốn API)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))  # Số docs giữ lại sau rerank cho LLM
    RERANKER_MAX_LENGTH: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))  # Tối ưu tốc độ suy luận Cross-Encoder trên CPU
    RERANKER_TIMEOUT: int = int(os.getenv("RERANKER_TIMEOUT", "25"))

    # -------------------------------------------------------------
    # 5. Preprocessing & Legal Chunker
    # -------------------------------------------------------------
    LEGAL_CHUNK_MAX_SIZE: int = int(os.getenv("LEGAL_CHUNK_MAX_SIZE", "1400"))
    LEGAL_CHUNK_MIN_SIZE: int = int(os.getenv("LEGAL_CHUNK_MIN_SIZE", "80"))

    # -------------------------------------------------------------
    # 6. System & Seed Parameters
    # -------------------------------------------------------------
    SEED: int = int(os.getenv("SEED", "42"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))

    # -------------------------------------------------------------
    # 7. Evaluation Parameters (Ragas Framework)
    # -------------------------------------------------------------
    _raw_eval_sample = os.getenv("EVAL_SAMPLE_SIZE", "")
    EVAL_SAMPLE_SIZE: Optional[int] = int(_raw_eval_sample) if _raw_eval_sample.isdigit() else None
    EVAL_MAX_WORKERS: int = int(os.getenv("EVAL_MAX_WORKERS", "2"))
    EVAL_TIMEOUT: int = int(os.getenv("EVAL_TIMEOUT", "180"))
    EVAL_MAX_RETRIES: int = int(os.getenv("EVAL_MAX_RETRIES", "2"))
    EVAL_MAX_WAIT: int = int(os.getenv("EVAL_MAX_WAIT", "60"))
    EVAL_STRICTNESS: int = int(os.getenv("EVAL_STRICTNESS", "1"))

    # -------------------------------------------------------------
    # 8. Web App & FastAPI Server
    # -------------------------------------------------------------
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    _raw_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
    ALLOWED_ORIGINS: List[str] = [orig.strip() for orig in _raw_allowed_origins.split(",") if orig.strip()]
    RATE_LIMIT_MAX: int = int(os.getenv("RATE_LIMIT_MAX", "20"))  # Số request tối đa
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Trong cửa sổ N giây

    @classmethod
    def set_global_seed(cls, seed_value: int = None):
        """Cố định seed ngẫu nhiên cho toàn bộ quy trình thực nghiệm"""
        target_seed = seed_value or cls.SEED
        random.seed(target_seed)
        np.random.seed(target_seed)
        os.environ["PYTHONHASHSEED"] = str(target_seed)
        print(f"[Settings] Global seed set to {target_seed}")


settings = Settings()
