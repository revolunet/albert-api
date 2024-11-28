from typing import List, Literal, Optional, Union

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, Field, model_validator

from app.utils.exceptions import ContextLengthExceededException, MaxTokensExceededException, WrongModelTypeException
from app.utils.lifespan import clients
from app.utils.variables import LANGUAGE_MODEL_TYPE, DEFAULT_RAG_TEMPLATE
from app.schemas.search import RagParameters


class ChatRagParameters(RagParameters):
    template: str = Field(description="Template to use for the RAG query", default=DEFAULT_RAG_TEMPLATE)


class ChatCompletionRequest(BaseModel):
    # See https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py
    messages: List[ChatCompletionMessageParam]
    model: str
    stream: Optional[Literal[True, False]] = False
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = Field(default_factory=list)
    user: Optional[str] = None
    best_of: Optional[int] = None
    top_k: int = -1
    min_p: float = 0.0

    # Albert additionnal fields
    rag: bool = False
    rag_parameters: ChatRagParameters = Field(default_factory=ChatRagParameters)

    class ConfigDict:
        extra = "allow"

    @model_validator(mode="after")
    def validate_model(cls, chat_completion_request):
        if clients.models[chat_completion_request.model].type != LANGUAGE_MODEL_TYPE:
            raise WrongModelTypeException()

        if "messages" in chat_completion_request and not clients.models[chat_completion_request.model].check_context_length(messages=chat_completion_request.messages):
            raise ContextLengthExceededException()

        if chat_completion_request.max_tokens is not None and chat_completion_request.max_tokens > clients.models[chat_completion_request.model].max_context_length:
            raise MaxTokensExceededException()

        return chat_completion_request


class ChatCompletion(ChatCompletion):
    pass


class ChatCompletionChunk(ChatCompletionChunk):
    pass
