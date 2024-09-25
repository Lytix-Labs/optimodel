import time
import uuid
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from typing import Dict, Any, List
import os
import json

from optimodel_server.GuardClient.guardClient import GuardClient
from optimodel_server.OptimodelError import OptimodelError
from optimodel_server.Routes import LytixProxyResponse
import logging
from optimodel_server.Config import config
import sys

from optimodel_server.Utils.QueryModelMain import (
    check_post_query_guards,
    check_pre_query_guards,
    queryModelMain,
)
from optimodel_types import (
    AnthropicCredentials,
    Credentials,
    GeminiCredentials,
    GroqCredentials,
    Guards,
    MistralAICredentials,
    MistralCodeStralCredentials,
    ModelTypes,
    QueryBody,
    TogetherAICredentials,
)
from optimodel_types.providerTypes import GuardError, MakeQueryResponse
from optimodel_types import (
    LLamaPromptGuardConfig,
    LytixRegexConfig,
    MicrosoftPresidioConfig,
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

openaiRouter = APIRouter()

OPENAI_API_URL = "https://api.openai.com/v1"

guardClientInstance = GuardClient(config.guardServerURL)


def create_guard(guard_dict):
    guard_type = guard_dict.get("guardName")
    if guard_type == "META_LLAMA_PROMPT_GUARD_86M":
        return LLamaPromptGuardConfig(**guard_dict)
    elif guard_type == "LYTIX_REGEX_GUARD":
        return LytixRegexConfig(**guard_dict)
    elif guard_type == "MICROSOFT_PRESIDIO_GUARD":
        return MicrosoftPresidioConfig(**guard_dict)
    else:
        raise ValueError(f"Unknown guard type: {guard_type}")


@openaiRouter.api_route("/{path:path}", methods=["POST"])
async def openai_chat_proxy(request: Request, path: str):
    # Get the request body as JSON
    body = await request.json()

    # Extract necessary parameters from the request body
    model = body.get("model")
    stream = body.get("stream", False)
    guards = body.get("lytix-guards", [])
    speedPriority = body.get("lytix-speedPriority", 0)
    # Rewrite since optimodel generally uses "guards" instead of "lytix-guards"
    body["speedPriority"] = speedPriority
    body["guards"] = guards
    messages = []

    """
    If we have any guards, split them up based on pre vs post query
    """
    preQueryGuards = []
    postQueryGuards = []
    if guards:
        for guard in guards:
            if guard["guardType"] == "preQuery":
                preQueryGuards.append(guard)
            if guard["guardType"] == "postQuery":
                postQueryGuards.append(guard)

    # # """
    # # Now attempt the query with each provider in order
    # # """
    # # errors = []
    # guardErrors: List[GuardError] = []

    try:
        # Parse user and system messages
        for message in body.get("messages", []):
            role = message.get("role")
            content = message.get("content")

            if isinstance(content, list):
                # If content is a list, extract text content
                text_content = next(
                    (item["content"] for item in content if item["type"] == "text"),
                    None,
                )
                if text_content:
                    messages.append(
                        {
                            "role": role,
                            "content": [{"type": "text", "text": text_content}],
                        }
                    )
            elif isinstance(content, str):
                # If content is a string, add it directly
                messages.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )
    except Exception as e:
        logger.error(f"Error attempting to extract messages", e)

    if model is None:
        raise OptimodelError("model is required")

    match model.lower().replace("-", "_"):
        case (
            ModelTypes.gpt_4.name
            | ModelTypes.gpt_3_5_turbo.name
            | ModelTypes.gpt_4o.name
            | ModelTypes.gpt_4_turbo.name
            | ModelTypes.gpt_3_5_turbo_0125.name
            | ModelTypes.gpt_4o_mini.name
            | ModelTypes.gpt_4o_mini_2024_07_18.name
            | ModelTypes.gpt_4o_2024_08_06.name
            | ModelTypes.o1_preview.name
            | ModelTypes.o1_preview_2024_09_12.name
            | ModelTypes.o1_mini.name
            | ModelTypes.o1_mini_2024_09_12.name
        ):
            """
            Nothing to do here, this is just the normal openai proxy
            """
            pass
        case _:
            """
            Based on the headers create our credentials object
            """
            credentials: list[Credentials] = []
            if "mistralapikey" in request.headers:
                credentials.append(
                    MistralAICredentials(mistralApiKey=request.headers["mistralapikey"])
                )
            if "anthropicapikey" in request.headers:
                credentials.append(
                    AnthropicCredentials(
                        anthropicApiKey=request.headers["anthropicapikey"]
                    )
                )
            if "groqapikey" in request.headers:
                credentials.append(
                    GroqCredentials(groqApiKey=request.headers["groqapikey"])
                )
            if "togetherapikey" in request.headers:
                credentials.append(
                    TogetherAICredentials(
                        togetherApiKey=request.headers["togetherapikey"]
                    )
                )
            if "geminiapikey" in request.headers:
                credentials.append(
                    GeminiCredentials(geminiApiKey=request.headers["geminiapikey"])
                )
            if "mistralcodestralapikey" in request.headers:
                credentials.append(
                    MistralCodeStralCredentials(
                        mistralCodeStralApiKey=request.headers["mistralcodestralapikey"]
                    )
                )

            body = QueryBody(
                messages=messages,
                modelToUse=model,
                credentials=credentials,
                guards=guards,
            )
            response = await queryModelMain(body, guardClientInstance)

            if isinstance(response, JSONResponse):
                return response

            responseParsed: MakeQueryResponse = response

            lytixProxyPayload: LytixProxyResponse | None = None
            try:
                """
                We need to get the lytix-proxy-payload from the response
                """
                lytixProxyPayload = {
                    "messagesV2": messages
                    + [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": responseParsed["modelResponse"],
                                }
                            ],
                        }
                    ],
                    "inputTokens": responseParsed["promptTokens"],
                    "outputTokens": responseParsed["generationTokens"],
                    "cost": responseParsed["cost"],
                    "provider": responseParsed["provider"],
                    "guardErrors": responseParsed["guardErrors"],
                }
                print(f">>lytixProxyPayload: {lytixProxyPayload['guardErrors'][1]}")
            except Exception as e:
                logger.error(f"Error attempting to extract lytix-proxy-payload", e)

            return Response(
                content=json.dumps(
                    {
                        "lytix-proxy-payload": lytixProxyPayload,
                        "id": str(uuid.uuid4()),
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "logprobs": None,
                                "message": {
                                    "content": responseParsed["modelResponse"],
                                    "role": "assistant",
                                },
                            },
                        ],
                        "created": time.time(),
                        "model": model,
                        "object": "chat.completion",
                        "service_tier": None,
                        "usage": {
                            "completion_tokens": responseParsed["generationTokens"],
                            "prompt_tokens": responseParsed["promptTokens"],
                            "total_tokens": responseParsed["promptTokens"]
                            + responseParsed["generationTokens"],
                        },
                    }
                ),
                status_code=200,
                media_type="application/json",
            )

    """
    If we have any guards, split them up based on pre vs post query
    every other model gets it inside queryModelMain
    """
    preQueryGuards = []
    postQueryGuards = []
    if guards:
        for guard in guards:
            if guard["guardType"] == "preQuery":
                preQueryGuards.append(guard)
            if guard["guardType"] == "postQuery":
                postQueryGuards.append(guard)

    guardErrorsFinal: List[GuardError] = []

    if preQueryGuards:
        guardErrors, should_return = await check_pre_query_guards(
            preQueryGuards=[create_guard(x) for x in preQueryGuards],
            guardClientInstance=guardClientInstance,
            messages=messages,
            providerName="openai",
        )
        if should_return:
            """
            GuardErrors here for some reason is a object that is the expected return
            TODO: This is horrible code but it works for now, refactor
            """
            return guardErrors
        if guardErrors:
            guardErrorsFinal.append(guardErrors)

    # Prepare the headers for the opeanai API call
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {request.headers.get('openaikey')}",
        **{
            k: v
            for k, v in request.headers.items()
            if k
            not in [
                "Content-Type",
                "authorization",
                "content-length",
                "host",
            ]
        },
    }

    # Remove any headers with None values
    headers = {k: str(v) for k, v in headers.items() if v is not None}

    # Construct the full path for the Anthropic API request
    full_url = f"{OPENAI_API_URL}/{path.lstrip('openai/')}"

    """
    Delete lytix-guards and guards, and lytix-speedPriority and speedPriority
    Otherwise openAI will complain
    """
    body = body.copy()
    body.pop("lytix-guards", None)
    body.pop("guards", None)
    body.pop("lytix-speedPriority", None)
    body.pop("speedPriority", None)

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

        lytixProxyPayload = None
        try:
            # Extract model messages
            if "choices" in response_data:
                for choice in response_data["choices"]:
                    if "message" in choice:
                        message = choice["message"]
                        if "content" in message:
                            try:
                                # Attempt to parse the content as JSON
                                content_json = json.loads(message["content"])
                                if "response" in content_json:
                                    messages.append(
                                        {
                                            "role": message["role"],
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": content_json["response"],
                                                }
                                            ],
                                        }
                                    )
                            except json.JSONDecodeError:
                                # If parsing fails, treat the content as plain text
                                messages.append(
                                    {
                                        "role": message["role"],
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": message["content"],
                                            }
                                        ],
                                    }
                                )

            # Extract token usage
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

            # Now attempt to calculate cost based on the model
            modelData = config.modelToProvider.get(
                model.lower().replace("-", "_"), None
            )

            # Get the first openai provider
            modelData = next((x for x in modelData if x["provider"] == "openai"), None)

            cost = None
            if modelData is not None:
                cost = modelData["pricePer1MInput"] * (
                    input_tokens / 1_000_000
                ) + modelData["pricePer1MOutput"] * (output_tokens / 1_000_000)
            lytixProxyPayload = LytixProxyResponse(
                messagesV2=messages,
                inputTokens=input_tokens,
                outputTokens=output_tokens,
                cost=cost,
                provider="openai",
            ).dict()
        except Exception as e:
            logger.error(f"Error attempting to extract usage tokens", e)

        if postQueryGuards:
            if messages is None:
                raise OptimodelError(
                    "messages are required for guard. Please reach out to support@lytix.co if you believe this is an error"
                )
            guardErrors, queryResponse = await check_post_query_guards(
                postQueryGuards=[create_guard(x) for x in postQueryGuards],
                guardClientInstance=guardClientInstance,
                messages=messages,
                modelOutput=response_data["choices"][0]["message"]["content"],
                queryResponse={},
            )
            if guardErrors:
                guardErrorsFinal.append(guardErrors)

        if guardErrorsFinal:
            lytixProxyPayload["guardErrors"] = guardErrorsFinal

        return Response(
            content=json.dumps(
                {
                    "lytix-proxy-payload": lytixProxyPayload,
                    **response_data,
                }
            ),
            status_code=response.status_code,
            media_type="application/json",
        )


@openaiRouter.api_route("/{path:path}", methods=["GET"])
async def openai_get_proxy(request: Request, path: str):
    """
    Blindly forward and get requests, dont intercept anything just proxy
    """
    # Construct the full URL for the Anthropic API request
    full_url = f"{OPENAI_API_URL}/{path}"

    # Prepare the headers for the Anthropic API call
    headers = {
        "X-API-Key": request.headers.get("openaiApiKey"),
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
