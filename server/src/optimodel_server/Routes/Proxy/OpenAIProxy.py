import time
import uuid
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from typing import Dict, Any, List
import os
import json
import brotli

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
    # extract the lytix ioeventid if present
    ioEventId = request.headers.get("x-lytix-io-event-id")

    # Get the request body as JSON
    body = await request.json()

    # Extract necessary parameters from the request body
    model = body.get("model")
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
        for message in body.get("messages", []):
            role = message.get("role")
            content = message.get("content")

            if isinstance(content, list):
                processed_content = []
                for item in content:
                    if item.get("type") == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
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
            elif isinstance(content, str):
                # If content is a string, add it directly
                messages.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )
    except Exception as e:
        logger.error(f"Error attempting to extract messages", e)

    if model is None:
        raise OptimodelError("model is required")

    for index, modelToTry in enumerate(allModelsToTry):
        logger.info(f"Trying model: {modelToTry}")
        try:
            match modelToTry.lower().replace("-", "_").replace(".", "_"):
                case (
                    ModelTypes.gpt_4.name
                    | ModelTypes.gpt_3_5_turbo.name
                    | ModelTypes.gpt_4o.name
                    | ModelTypes.gpt_4_turbo.name
                    | ModelTypes.gpt_3_5_turbo_0125.name
                    | ModelTypes.gpt_4o_mini.name
                    | ModelTypes.gpt_4o_mini_2024_07_18.name
                    | ModelTypes.gpt_4o_2024_08_06.name
                    | ModelTypes.gpt_4o_2024_05_13.name
                    | ModelTypes.o1_preview.name
                    | ModelTypes.o1_preview_2024_09_12.name
                    | ModelTypes.o1_mini.name
                    | ModelTypes.o1_mini_2024_09_12.name
                ):
                    """
                    Nothing to do here, this is just the normal openai proxy
                    """
                    pass
                case model if "gpt" in model.lower():
                    """
                    Handle any GPT model that wasn't explicitly listed above
                    """
                    logger.info(f"Handling unlisted GPT model: {model}")
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
                            "lytixEventId": ioEventId,
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
                    providerName="openai",
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
                        provider="openai",
                        guardErrors=(
                            guardErrors["guardErrors"]
                            if "guardErrors" in guardErrors
                            else guardErrors
                        ),
                        lytixEventId=ioEventId,
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

            # Prepare the headers for the opeanai API call
            headers = {
                "content-type": "application/json",
                "accept": "application/json",
                "Authorization": f"Bearer {request.headers.get('openaikey')}",
                **{
                    k: v
                    for k, v in request.headers.items()
                    if k.lower()
                    not in [
                        "authorization",
                        "content-length",
                        "host",
                        "content-type",
                        "x-lytix-io-event-id",
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

                    # print(f">>> response: {response}")
                    # import pdb

                    # pdb.set_trace()

                    content_type = response.headers.get("Content-Type", "")
                    content_encoding = response.headers.get("Content-Encoding", "")

                    print(f">>> content_type: {content_type}")
                    if "br" in content_encoding.lower():
                        try:
                            # decompressed_data = brotli.decompress(response.content)
                            decompressed_data = response.content
                            response_data = json.loads(
                                decompressed_data.decode("utf-8")
                            )
                        except Exception as e:
                            print(
                                f"Failed to decompress or parse brotli-compressed content: {e}"
                            )
                            print(f"bytes of compressed content: {response.content}")
                            raise OptimodelError(
                                "Failed to process brotli-compressed response from OpenAI API"
                            )
                    elif "application/json" in content_type:
                        try:
                            response_data = response.json()
                        except json.JSONDecodeError:
                            raise OptimodelError(
                                "Failed to decode JSON response from OpenAI API"
                            )
                    else:
                        logger.warning(f"Unexpected response format from OpenAI API")
                        response_data = None
                        # raise OptimodelError(
                        #     f"Unexpected response format from OpenAI API"
                        # )

                # If its a non-200 response
                if response.status_code != 200:
                    raise OptimodelError(
                        f"Non-200 response from OpenAI API: {response.status_code}"
                    )

                lytixProxyPayload = None
                try:
                    if response_data is None:
                        continue
                    # Extract model messages
                    elif "choices" in response_data:
                        for choice in response_data["choices"]:
                            if "message" in choice:
                                message = choice["message"]
                                if "content" in message:
                                    messages.append(
                                        {
                                            "role": "assistant",
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
                        modelToTry.lower().replace("-", "_").replace(".", "_"), None
                    )

                    # Get the first openai provider
                    modelData = next(
                        (x for x in modelData if x["provider"] == "openai"), None
                    )

                    cost = None
                    if modelData is not None:
                        cost = modelData["pricePer1MInput"] * (
                            input_tokens / 1_000_000
                        ) + modelData["pricePer1MOutput"] * (output_tokens / 1_000_000)
                    lytixProxyPayload = LytixProxyResponse(
                        lytixEventId=ioEventId,
                        messagesV2=messages,
                        inputTokens=input_tokens,
                        outputTokens=output_tokens,
                        cost=cost,
                        provider="openai",
                        model=modelToTry,
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
            logger.error(f"Error attempting to process openai request", e)

            """
            Unless we are out of models, continue
            """
            if index == len(allModelsToTry) - 1:
                logger.error(f"No more fallback models to try")
                if len(fallbackModels) > 0:
                    raise OptimodelError("No more fallback models to try")
                else:
                    raise e


@openaiRouter.api_route("/{path:path}", methods=["GET"])
async def openai_get_proxy(
    request: Request, path: str
):  # extract the lytix ioeventid if present
    ioEventId = request.headers.get("x-lytix-io-event-id")
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
