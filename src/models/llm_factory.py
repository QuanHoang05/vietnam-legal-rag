"""
Factory tạo đối tượng LLM kết nối OpenRouter linh hoạt.
Cho phép chuyển đổi qua lại giữa các mô hình lớn khác nhau
(GPT-4o-mini, Claude 3.5 Sonnet, DeepSeek V3, LLaMA 3, Qwen 2.5...)
thông qua API Gateway duy nhất của OpenRouter.
"""

from typing import Optional, Any
from configs.settings import settings



def get_openrouter_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Khởi tạo ChatOpenAI cấu hình tới OpenRouter endpoint.
    
    Args:
        model_name: Mã định danh mô hình trên OpenRouter (VD: 'openai/gpt-4o-mini', 'deepseek/deepseek-chat')
        temperature: Nhiệt độ sinh (mặc định 0.0 để đảm bảo tính xác định)
        max_tokens: Số token tối đa được sinh ra
        seed: Seed để tái lập kết quả
        api_key: API Key OpenRouter (nếu None sẽ đọc từ settings)
        base_url: URL base của OpenRouter (mặc định: https://openrouter.ai/api/v1)
    """
    actual_model = model_name or settings.OPENROUTER_MODEL
    actual_key = api_key or settings.OPENROUTER_API_KEY
    actual_base_url = base_url or settings.OPENROUTER_BASE_URL
    actual_temp = settings.TEMPERATURE if temperature is None else temperature
    actual_max_tokens = max_tokens or settings.MAX_TOKENS
    actual_seed = seed if seed is not None else settings.SEED

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

    # ChatOpenAI tương thích hoàn toàn với OpenRouter
    # request_timeout=60: tránh mặc định 600s × 3 retry treo threadpool (M4)
    llm = ChatOpenAI(
        model=actual_model,
        temperature=actual_temp,
        max_tokens=actual_max_tokens,
        seed=actual_seed,
        api_key=actual_key,
        base_url=actual_base_url,
        default_headers=default_headers,
        request_timeout=60,
    )
    return llm


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Hàm khởi tạo LLM tương thích tài liệu baseline và nâng cao.
    Mặc định định tuyến qua OpenRouter để người dùng đổi model linh hoạt.
    """
    return get_openrouter_llm(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        api_key=api_key,
        base_url=base_url,
    )
