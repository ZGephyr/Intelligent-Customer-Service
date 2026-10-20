"""对话模型适配层：LangChain + OpenAI 兼容协议。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger

logger = get_logger(__name__)


class ChatModelProvider(Protocol):
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...

    def invoke(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str: ...


class OpenAIChatProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout_seconds: float = 60,
        client: Any | None = None,
    ) -> None:
        self._model_name = model
        self._client = client or ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=1,
        )

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        try:
            for chunk in self._client.stream(messages):
                text = getattr(chunk, "content", "")
                if text:
                    yield text
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("模型流式生成失败: model=%s", self._model_name)
            raise AppError(ErrorCode.LLM_ERROR, f"模型调用失败: {exc}") from exc

    def invoke(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        try:
            response = self._client.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("模型调用失败: model=%s", self._model_name)
            raise AppError(ErrorCode.LLM_ERROR, f"模型调用失败: {exc}") from exc
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content)


def build_chat_provider(settings, timeout_seconds: float | None = None) -> OpenAIChatProvider:
    return OpenAIChatProvider(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=timeout_seconds or settings.llm_timeout_seconds,
    )
