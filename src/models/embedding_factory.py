"""
Embedding Model cục bộ (Local Embedding) sử dụng BAAI/bge-m3.
- Chạy hoàn toàn trên máy cục bộ bằng HuggingFace / SentenceTransformers.
- 100% Offline, không phụ thuộc API key, không tốn chi phí.
- Không giới hạn Rate Limit (0đ, không phụ thuộc mạng).
- Hỗ trợ tiếng Việt và đa ngữ vượt trội với 1024 dimensions.
- Cấu hình linh hoạt qua configs.settings.
"""

from typing import List, Optional
from langchain_core.embeddings import Embeddings
from configs.settings import settings


class BGEM3Embeddings(Embeddings):
    """
    Embedding Client cục bộ chuẩn BAAI/bge-m3 qua sentence-transformers.
    Tự động chuẩn hóa văn bản, chạy theo batch và tính toán trực tiếp trên CPU/GPU máy cá nhân.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Chưa cài đặt thư viện 'sentence-transformers'. Vui lòng chạy: pip install sentence-transformers"
            )

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[BGEM3Embeddings] Khởi tạo mô hình '{self.model_name}' trên thiết bị: {self.device} (batch_size={self.batch_size})...")
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Nhúng danh sách văn bản thành danh sách vector (1024 chiều)"""
        if not texts:
            return []

        cleaned = [t.strip() if (t and t.strip()) else "đoạn trống" for t in texts]
        embeddings = self._model.encode(
            cleaned,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Nhúng một câu truy vấn thành vector"""
        clean = text.strip() if (text and text.strip()) else "đoạn trống"
        embedding = self._model.encode(
            clean,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding.tolist()


# Singleton instance để tránh load model nhiều lần vào RAM
_GLOBAL_EMBEDDING_INSTANCE: Optional[BGEM3Embeddings] = None


def get_embeddings(
    model_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    _provider: Optional[str] = None,
) -> Embeddings:
    """
    Khởi tạo hoặc tái sử dụng instance Embedding BAAI/bge-m3 cục bộ.
    Đọc cấu hình mặc định từ settings.EMBEDDING_MODEL và settings.EMBEDDING_BATCH_SIZE.
    """
    global _GLOBAL_EMBEDDING_INSTANCE
    target_model = model_name or settings.EMBEDDING_MODEL
    target_batch = chunk_size or settings.EMBEDDING_BATCH_SIZE

    if _GLOBAL_EMBEDDING_INSTANCE is None or _GLOBAL_EMBEDDING_INSTANCE.model_name != target_model:
        _GLOBAL_EMBEDDING_INSTANCE = BGEM3Embeddings(
            model_name=target_model,
            batch_size=target_batch,
        )
    return _GLOBAL_EMBEDDING_INSTANCE
