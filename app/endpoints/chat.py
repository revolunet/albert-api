from typing import Union

from fastapi import APIRouter, Request, Security
from fastapi.responses import StreamingResponse
import httpx

from app.schemas.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from app.schemas.security import User
from app.utils.settings import settings
from app.utils.lifespan import clients, limiter
from app.utils.security import check_api_key, check_rate_limit

router = APIRouter()


@router.post("/chat/completions")
@limiter.limit(settings.default_rate_limit, key_func=lambda request: check_rate_limit(request=request))
async def chat_completions(
    request: Request, body: ChatCompletionRequest, user: User = Security(check_api_key)
) -> Union[ChatCompletion, ChatCompletionChunk]:
    """Completion API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/chat/create for the API specification.
    """
    client = clients.models[body.model]
    url = f"{client.base_url}chat/completions"
    headers = {"Authorization": f"Bearer {client.api_key}"}

    # non stream case
    if not body.stream:
        async with httpx.AsyncClient(timeout=20) as async_client:
            response = await async_client.request(method="POST", url=url, headers=headers, json=body.model_dump())
            response.raise_for_status()

            data = response.json()
            return ChatCompletion(**data)

    # stream case
    async def forward_stream(url: str, headers: dict, request: dict):
        async with httpx.AsyncClient(timeout=20) as async_client:
            async with async_client.stream(method="POST", url=url, headers=headers, json=request) as response:
                async for chunk in response.aiter_raw():
                    yield chunk

    return StreamingResponse(forward_stream(url, headers, body.model_dump()), media_type="text/event-stream")
