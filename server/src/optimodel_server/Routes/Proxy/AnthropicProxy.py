from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
import httpx
from typing import Dict, Any, List
import os
import json

from optimodel_server.Config import config
from optimodel_server.OptimodelError import OptimodelError
from optimodel_server.Routes import LytixProxyResponse
from optimodel_types import ModelMessage

anthropicRouter = APIRouter()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1"


@anthropicRouter.api_route("/{path:path}", methods=["GET"])
async def anthropic_get_proxy(request: Request, path: str):
    """
    Blindly forward and get requests, dont intercept anything just proxy
    """
    # Construct the full URL for the Anthropic API request
    full_url = f"{ANTHROPIC_API_URL}/{path.lstrip('anthropic/')}"

    # Prepare the headers for the Anthropic API call
    headers = {
        "X-API-Key": request.headers.get("anthropicApiKey"),
        "anthropic-version": "2023-06-01",
        **{
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length", "x-lytix-io-event-id"]
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
    # extract the lytix ioeventid if present
    ioEventId = request.headers.get("x-lytix-io-event-id")

    # Get the request body as JSON
    body = await request.json()
    messages: List[ModelMessage] = []

    # Extract necessary parameters from the request body
    model = body.get("model")
    stream = body.get("stream", False)

    if model is None:
        raise OptimodelError("model is required")

    # Prepare the headers for the Anthropic API call
    headers = {
        "x-api-key": request.headers.get("anthropicApiKey"),
        **{
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in [
                "x-api-key",
                "content-length",
                "host",
                "x-lytix-io-event-id",
            ]
        },
    }

    # Extract all role content from messages
    # Handle system messages
    system_messages = body.get("system", [])
    if isinstance(system_messages, list):
        for message in system_messages:
            if isinstance(message, dict) and message.get("type") == "text":
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": message["text"],
                            }
                        ],
                    }
                )
    elif isinstance(system_messages, str):
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_messages,
                    }
                ],
            }
        )

    for message in body.get("messages", []):
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content["type"] == "text":
                    messages.append(
                        {
                            "role": message.get("role"),
                            "content": [
                                {
                                    "type": "text",
                                    "text": content["text"],
                                }
                            ],
                        }
                    )
        elif isinstance(message.get("content"), str):
            messages.append(
                {
                    "role": message.get("role"),
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"],
                        }
                    ],
                }
            )

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
            response = await client.post(
                full_url, json=body, headers=headers, timeout=600
            )

        response_data = response.json()
        # Extract content from response_data
        if isinstance(response_data.get("content"), list):
            for content in response_data["content"]:
                if content["type"] == "text":
                    messages.append(
                        {
                            "role": response_data.get("role"),
                            "content": [
                                {
                                    "type": "text",
                                    "text": content["text"],
                                }
                            ],
                        }
                    )
        elif isinstance(response_data.get("content"), str):
            messages.append(
                {
                    "role": response_data.get("role"),
                    "type": "text",
                    "content": [
                        {
                            "type": "text",
                            "text": response_data["content"],
                        }
                    ],
                }
            )

        # Extract model parameters
        try:
            # Extract token usage
            # response_data: {'id': 'msg_01WSAwZBmXhafnXHRTe5oekP', 'type': 'message', 'role': 'assistant', 'model': 'claude-3-haiku-20240307', 'content': [{'type': 'text', 'text': 'Hello! I\'m Claude, an AI assistant created by Anthropic. I don\'t have a name like "beb87629-4be3-457a-9ee0-21c96c0c2a13" - that looks like a unique identifier rather than a name. How can I assist you today?'}], 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 38, 'output_tokens': 71}}
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            print(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}")

            # Now attempt to calculate cost based on the model
            modelData = config.modelToProvider.get(
                model.lower().replace("-", "_").replace(".", "_"), None
            )

            # Get the first anthropic provider
            modelData = next(
                (x for x in modelData if x["provider"] == "anthropic"), None
            )

            cost = None
            if (
                modelData is not None
                and input_tokens is not None
                and output_tokens is not None
            ):
                cost = modelData["pricePer1MInput"] * (
                    input_tokens / 1_000_000
                ) + modelData["pricePer1MOutput"] * (output_tokens / 1_000_000)

        except Exception as e:
            print(f"Error attempting to calculate cost", e)

        # print(f"cost: {cost}, tokens: {input_tokens}, {output_tokens}")
        return Response(
            content=json.dumps(
                {
                    "lytix-proxy-payload": LytixProxyResponse(
                        lytixEventId=ioEventId,
                        messagesV2=messages,
                        inputTokens=input_tokens,
                        outputTokens=output_tokens,
                        cost=cost,
                        provider="anthropic",
                        model=model,
                    ).dict(),
                    **response.json(),
                }
            ),
            status_code=response.status_code,
            media_type="application/json",
        )
