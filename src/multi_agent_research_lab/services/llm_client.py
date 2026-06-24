"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AgentExecutionError("OPENAI_API_KEY is not set in .env")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._timeout = settings.timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry on transient errors."""

        logger.debug("LLM request | model=%s", self._model)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self._timeout,
        )
        choice = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        logger.debug("LLM response | in=%s out=%s tokens", input_tokens, output_tokens)
        return LLMResponse(
            content=choice,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
