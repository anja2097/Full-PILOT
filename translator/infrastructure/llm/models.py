"""Modelos Pydantic para la respuesta de la API de OpenRouter (formato OpenAI)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReasoningDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[ReasoningDetail] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: Message
    finish_reason: str | None = None


class PromptTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int = 0


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None
    cost: float | None = None


class ChatCompletion(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class ApiError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: int | str
    message: str
    metadata: dict | None = None


class ErrorResponse(BaseModel):
    error: ApiError
