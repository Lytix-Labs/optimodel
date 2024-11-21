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
import re


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

geminiRouter = APIRouter()

GEMINI_API_URL = "https://generativelanguage.googleapis.com"


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


@geminiRouter.api_route("/{path:path}", methods=["POST"])
async def gemini_chat_proxy(request: Request, path: str):
    # Get the request body as JSON
    body = await request.json()

    # Extract necessary parameters from the request body
    # model = body.get("model")
    # print(">>>>>", json.dumps(body, indent=2))

    model_match = re.search(r"/v1beta/models/([^:]+)", path)
    model = model_match.group(1) if model_match else None
    stream = body.get("stream", False)
    guards = body.get("lytix-guards", [])
    fallbackModels = body.get("lytix-fallbackModels", [])
    speedPriority = body.get("lytix-speedPriority", 0)
    provider = body.get("lytix-provider", None)

    # Rewrite since optimodel generally uses "guards" instead of "lytix-guards"
    body["speedPriority"] = speedPriority
    body["guards"] = guards
    body["fallbackModels"] = fallbackModels
    body["provider"] = provider
    messages = []

    """
    Define all models to try, its our main model + any fallbacks
    """
    allModelsToTry = [model, *fallbackModels]

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

    """
    Keep track of what models have failed, along with their error message
    @TODO this
    """

    try:
        # Parse user and system messages
        for content in body.get("contents", []):
            role = content.get("role")
            parts = content.get("parts", [])

            processed_content = []
            for part in parts:
                if "text" in part:
                    processed_content.append({"type": "text", "text": part["text"]})
                elif "image_url" in part:
                    image_url = part["image_url"]["url"]
                    if image_url.startswith("data:"):
                        # Parse the data URL
                        _, data = image_url.split(",", 1)
                        media_type = image_url.split(";")[0].split(":")[1]
                        processed_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "mediaType": media_type,
                                    "data": data,
                                },
                            }
                        )
        if processed_content:
            messages.append({"role": role, "content": processed_content})

        """
        System message is next to model
        """
        # Extract system message from the body
        system_instruction = body.get("systemInstruction", {})
        if system_instruction:
            system_message = {
                "role": system_instruction.get("role", "system"),
                "content": [
                    {
                        "type": "text",
                        "text": system_instruction.get("parts", [{}])[0].get(
                            "text", "You are a helpful assistant."
                        ),
                    }
                ],
            }
            messages.append(system_message)

    except Exception as e:
        logger.error(f"Error attempting to extract messages", e)

    if model is None:
        raise OptimodelError("model is required")

    for index, modelToTry in enumerate(allModelsToTry):
        try:
            match modelToTry.lower().replace("-", "_").replace(".", "_"):
                case (
                    ModelTypes.gemini_1_5_pro.name
                    | ModelTypes.gemini_1_5_pro_001.name
                    | ModelTypes.gemini_1_5_pro_exp_0801
                    | ModelTypes.gemini_1_5_pro_exp_0827.name
                    | ModelTypes.gemini_1_5_flash.name
                    | ModelTypes.gemini_1_5_flash_latest.name
                    | ModelTypes.gemini_1_5_flash_001.name
                    | ModelTypes.gemini_1_5_flash_001_tuning.name
                    | ModelTypes.gemini_1_5_flash_exp_0827.name
                    | ModelTypes.gemini_1_5_flash_8b_exp_0827.name
                    | ModelTypes.gemini_1_5_pro_latest.name
                    | ModelTypes.gemini_1_5_pro_002.name
                    | ModelTypes.gemini_1_5_flash_8b.name
                ):
                    """
                    Nothing to do here, this is just the normal gemini proxy
                    """
                    pass
                case model if "gemini" in model:
                    """
                    Catch all, if any have the word gemini in them, we need to use the new gemini proxy
                    """
                    pass
                case _:
                    """
                    Based on the headers create our credentials object
                    """
                    credentials: list[Credentials] = []
                    if "mistralapikey" in request.headers:
                        credentials.append(
                            MistralAICredentials(
                                mistralApiKey=request.headers["mistralapikey"]
                            )
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
                            GeminiCredentials(
                                geminiApiKey=request.headers["geminiapikey"]
                            )
                        )
                    if "mistralcodestralapikey" in request.headers:
                        credentials.append(
                            MistralCodeStralCredentials(
                                mistralCodeStralApiKey=request.headers[
                                    "mistralcodestralapikey"
                                ]
                            )
                        )

                    body = QueryBody(
                        messages=messages,
                        modelToUse=modelToTry,
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
                    except Exception as e:
                        logger.error(
                            f"Error attempting to extract lytix-proxy-payload", e
                        )

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
                                "model": modelToTry,
                                "object": "chat.completion",
                                "service_tier": None,
                                "usage": {
                                    "completion_tokens": responseParsed[
                                        "generationTokens"
                                    ],
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
                    providerName="gemini",
                )
                if should_return:
                    """
                    GuardErrors here for some reason is a object that is the expected return
                    TODO: This is horrible code but it works for now, refactor
                    """
                    if guardErrors["modelResponse"] is not None:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": guardErrors["modelResponse"],
                                    }
                                ],
                            }
                        )
                    lytixProxyPayload = LytixProxyResponse(
                        messagesV2=messages,
                        inputTokens=0,
                        outputTokens=0,
                        cost=0,
                        provider="gemini",
                        guardErrors=(
                            guardErrors["guardErrors"]
                            if "guardErrors" in guardErrors
                            else guardErrors
                        ),
                    ).dict()

                    return {
                        "lytix-proxy-payload": lytixProxyPayload,
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "logprobs": None,
                                "message": {
                                    "content": guardErrors["modelResponse"],
                                    "role": "assistant",
                                },
                            }
                        ],
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                            "total_tokens": 0,
                        },
                    }

                if guardErrors:
                    guardErrorsFinal.extend(guardErrors)

            # Prepare the headers for the gemini API call
            headers = {
                **{
                    k: v
                    for k, v in request.headers.items()
                    if k
                    not in [
                        "Content-Type",
                        "content-length",
                        "host",
                    ]
                },
            }

            # Remove any headers with None values
            headers = {k: str(v) for k, v in headers.items() if v is not None}

            # Construct the full path for the Anthropic API request
            full_url = f"{GEMINI_API_URL}/{path.lstrip('gemini/')}"

            """
            Delete lytix-guards and guards, and lytix-speedPriority and speedPriority
            Otherwise gemini will complain
            """
            body = body.copy()
            body.pop("lytix-guards", None)
            body.pop("guards", None)
            body.pop("lytix-speedPriority", None)
            body.pop("speedPriority", None)
            body.pop("lytix-fallbackModels", None)
            body.pop("fallbackModels", None)
            body.pop("lytix-provider", None)
            body.pop("provider", None)

            """
            Redefine model field
            """
            body["model"] = modelToTry

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

                lytixProxyPayload = None
                try:
                    # Extract model messages
                    if "candidates" in response_data:
                        for candidate in response_data["candidates"]:
                            if "content" in candidate:
                                content = candidate["content"]
                                if "parts" in content:
                                    for part in content["parts"]:
                                        if "text" in part:
                                            messages.append(
                                                {
                                                    "role": "assistant",
                                                    "content": [
                                                        {
                                                            "type": "text",
                                                            "text": part["text"],
                                                        }
                                                    ],
                                                }
                                            )

                    # Extract token usage
                    usage = response_data.get("usageMetadata", {})
                    input_tokens = usage.get("promptTokenCount")
                    output_tokens = usage.get("candidatesTokenCount")

                    # Now attempt to calculate cost based on the model
                    modelData = config.modelToProvider.get(
                        modelToTry.lower().replace("-", "_").replace(".", "_"), None
                    )

                    # Get the first gemini provider
                    try:
                        modelData = next(
                            (x for x in modelData if x["provider"] == "gemini"), None
                        )

                        cost = None
                        if modelData is not None:
                            """
                            Attempt to pull out data from the file Uris
                            """
                            cost = modelData["pricePer1MInput"] * (
                                input_tokens / 1_000_000
                            ) + modelData["pricePer1MOutput"] * (
                                output_tokens / 1_000_000
                            )
                        lytixProxyPayload = LytixProxyResponse(
                            messagesV2=messages,
                            model=modelToTry,
                            inputTokens=input_tokens,
                            outputTokens=output_tokens,
                            cost=cost,
                            provider="gemini",
                        ).dict()
                    except Exception as e:
                        logger.error(
                            f"Error attempting to extract usage tokens: {e}, modelToTry={modelToTry}"
                        )
                except Exception as e:
                    logger.error(f"Error attempting to extract usage tokens: {e}")

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
                        guardErrorsFinal.extend(guardErrors)

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

        except Exception as e:
            logger.error(
                f"Error attempting to process gemini request: {e}, modelToTry={modelToTry}"
            )

            """
            Unless we are out of models, continue
            """
            if index == len(allModelsToTry) - 1:
                logger.error(f"No more fallback models to try")
                if len(fallbackModels) > 0:
                    raise OptimodelError("No more fallback models to try")
                else:
                    raise e


@geminiRouter.api_route("/{path:path}", methods=["GET"])
async def gemini_get_proxy(request: Request, path: str):
    """
    Blindly forward and get requests, dont intercept anything just proxy
    """
    # Construct the full URL for the Anthropic API request
    full_url = f"{GEMINI_API_URL}/{path}"

    # Prepare the headers for the Anthropic API call
    headers = {
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
