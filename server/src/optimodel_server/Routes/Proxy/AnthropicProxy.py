from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
import httpx
from typing import Dict, Any
import os
import json

from optimodel_server.OptimodelError import OptimodelError
from optimodel_server.Routes import LytixProxyResponse

anthropicRouter = APIRouter()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1"


@anthropicRouter.api_route("/{path:path}", methods=["GET"])
async def anthropic_get_proxy(request: Request, path: str):
    """
    Blindly forward and get requests, dont intercept anything just proxy
    """
    # Construct the full URL for the Anthropic API request
    full_url = f"{ANTHROPIC_API_URL}/{path}"

    # Prepare the headers for the Anthropic API call
    headers = {
        "X-API-Key": request.headers.get("anthropicApiKey"),
        "anthropic-version": "2023-06-01",
        **{
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length"]
        },
    }

    # Remove any headers with None values
    headers = {k: v for k, v in headers.items() if v is not None}

    # Get query parameters
    params = dict(request.query_params)

    async with httpx.AsyncClient() as client:
        response = await client.get(full_url, headers=headers, params=params)

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )


@anthropicRouter.api_route("/{path:path}", methods=["POST"])
async def anthropic_chat_proxy(request: Request, path: str):
    # Get the request body as JSON
    body = await request.json()

    # Extract necessary parameters from the request body
    model = body.get("model")
    stream = body.get("stream", False)

    if model is None:
        raise OptimodelError("model is required")

    # Prepare the headers for the Anthropic API call
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": request.headers.get("anthropicApiKey"),
        "anthropic-version": "2023-06-01",
        **{
            k: v
            for k, v in request.headers.items()
            if k not in ["Content-Type", "X-API-Key", "anthropic-version"]
        },
    }

    # Remove any headers with None values
    headers = {k: str(v) for k, v in headers.items() if v is not None}

    # Construct the full path for the Anthropic API request
    full_url = f"{ANTHROPIC_API_URL}/{path.lstrip('anthropic/')}"

    async def event_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", full_url, json=body, headers=headers
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    if stream:
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            response = await client.post(full_url, json=body, headers=headers)

        response_data = response.json()

        # Extract model messages
        messages = []
        if "content" in response_data:
            for content in response_data["content"]:
                if content["type"] == "text":
                    messages.append(
                        {"role": response_data["role"], "text": content["text"]}
                    )

        # Extract model parameters

        # Extract token usage
        usage = response_data.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        return Response(
            content=json.dumps(
                {
                    "lytix-proxy-payload": LytixProxyResponse(
                        messages=messages,
                        inputTokens=input_tokens,
                        outputTokens=output_tokens,
                    ).dict(),
                    **response.json(),
                }
            ),
            status_code=response.status_code,
            media_type="application/json",
        )
