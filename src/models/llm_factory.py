"""
Factory tạo đối tượng LLM kết nối OpenRouter linh hoạt.
Cho phép chuyển đổi qua lại giữa các mô hình lớn khác nhau
(GPT-4o-mini, Claude 3.5 Sonnet, DeepSeek V3, LLaMA 3, Qwen 2.5...)
thông qua API Gateway duy nhất của OpenRouter.
"""

from typing import Optional, Any, List
from configs.settings import settings


def get_openrouter_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Any:
    """
    Khởi tạo ChatOpenAI cấu hình tới OpenRouter endpoint.
    """
    actual_model = model_name or settings.OPENROUTER_MODEL
    actual_key = api_key or settings.OPENROUTER_API_KEY
    actual_base_url = base_url or settings.OPENROUTER_BASE_URL
    actual_temp = settings.TEMPERATURE if temperature is None else temperature
    actual_max_tokens = max_tokens or settings.MAX_TOKENS
    actual_seed = seed if seed is not None else settings.SEED
    actual_timeout = timeout or settings.LLM_TIMEOUT

    default_headers = {
        "HTTP-Referer": settings.OPENROUTER_REFERER,
        "X-Title": settings.OPENROUTER_TITLE,
    }

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "Chưa cài đặt thư viện 'langchain-openai'. Vui lòng chạy: pip install -r requirements.txt"
        )

    llm = ChatOpenAI(
        model=actual_model,
        temperature=actual_temp,
        max_tokens=actual_max_tokens,
        seed=actual_seed,
        api_key=actual_key,
        base_url=actual_base_url,
        default_headers=default_headers,
        request_timeout=actual_timeout,
    )
    return llm


def get_evaluator_llm() -> Any:
    """
    Khởi tạo LLM chuyên dùng cho việc đánh giá Ragas.
    Sử dụng trực tiếp ChatOpenAI tới Google Gemini OpenAI-compatible endpoint
    (đảm bảo có thuộc tính .temperature mà Ragas yêu cầu).
    """
    from langchain_openai import ChatOpenAI
    gemini_key = settings.GEMINI_API_KEY
    if gemini_key and len(gemini_key) > 10:
        return ChatOpenAI(
            model=settings.GEMINI_MODEL,
            api_key=gemini_key,
            base_url=settings.GEMINI_BASE_URL,
            temperature=0.0,
            request_timeout=settings.LLM_TIMEOUT,
        )
    return get_openrouter_llm(model_name=settings.OPENROUTER_MODEL, temperature=0.0)


class CascadeChatModel:
    """
    Bộ bọc LLM hỗ trợ tự động chuyển đổi mô hình (Cascade Fallback)
    giữa Google Gemini (miễn phí, nhanh) và OpenRouter free router.
    Tương thích hoàn toàn với LangChain BatchRAG và Ragas.
    """
    def __init__(self, primary: Any, fallbacks: Optional[List[Any]] = None):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.temperature = getattr(primary, "temperature", 0.0)

    def invoke(self, *args, **kwargs):
        models = [self.primary] + self.fallbacks
        last_exc = None
        for m in models:
            try:
                return m.invoke(*args, **kwargs)
            except Exception as e:
                last_exc = e
                continue
        raise last_exc

    def batch(self, *args, **kwargs):
        models = [self.primary] + self.fallbacks
        last_exc = None
        for m in models:
            try:
                return m.batch(*args, **kwargs)
            except Exception as e:
                last_exc = e
                continue
        raise last_exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Any:
    """
    Khởi tạo LLM với cơ chế Cascade Auto-Fallback đa tầng:
    1. Ưu tiên Google Gemini (cấu hình trong settings: GEMINI_MODEL và GEMINI_FALLBACK_MODELS)
    2. Fallback sang OpenRouter (cấu hình trong settings: OPENROUTER_MODEL)
    3. Fallback tiếp sang OPENROUTER_FALLBACK_MODEL (mặc định openai/gpt-4o-mini)
    """
    from langchain_openai import ChatOpenAI

    candidates = []
    actual_timeout = timeout or settings.LLM_TIMEOUT
    actual_temp = temperature if temperature is not None else settings.TEMPERATURE

    # 1. Google Gemini qua OpenAI-compatible endpoint
    gemini_key = api_key if (base_url and "generativelanguage" in base_url) else settings.GEMINI_API_KEY
    if gemini_key and len(gemini_key) > 10:
        gemini_models = [settings.GEMINI_MODEL]
        for fb in settings.GEMINI_FALLBACK_MODELS:
            if fb not in gemini_models:
                gemini_models.append(fb)

        for gm in gemini_models:
            candidates.append(
                ChatOpenAI(
                    model=gm,
                    api_key=gemini_key,
                    base_url=settings.GEMINI_BASE_URL,
                    temperature=actual_temp,
                    request_timeout=actual_timeout,
                )
            )

    # 2. OpenRouter auto-router & fallback models
    if settings.OPENROUTER_API_KEY and "your" not in settings.OPENROUTER_API_KEY:
        target_model = model_name or settings.OPENROUTER_MODEL
        candidates.append(
            get_openrouter_llm(
                model_name=target_model,
                temperature=actual_temp,
                max_tokens=max_tokens,
                seed=seed,
                timeout=actual_timeout,
            )
        )
        if target_model != settings.OPENROUTER_FALLBACK_MODEL:
            candidates.append(
                get_openrouter_llm(
                    model_name=settings.OPENROUTER_FALLBACK_MODEL,
                    temperature=actual_temp,
                    max_tokens=max_tokens,
                    seed=seed,
                    timeout=actual_timeout,
                )
            )

    if not candidates:
        return get_openrouter_llm(
            model_name=model_name,
            temperature=actual_temp,
            max_tokens=max_tokens,
            seed=seed,
            api_key=api_key,
            base_url=base_url,
            timeout=actual_timeout,
        )

    primary = candidates[0]
    fallbacks = candidates[1:]
    if fallbacks:
        return CascadeChatModel(primary=primary, fallbacks=fallbacks)
    return primary
