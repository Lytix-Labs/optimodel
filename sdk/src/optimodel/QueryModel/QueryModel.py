import asyncio
import json
import logging
from typing import Callable
import aiohttp

from optimodel.envVars import LytixCreds
from optimodel_server_types import (
    Credentials,
    Guards,
    ModelMessage,
    SpeedPriority,
    ModelTypes,
    Providers,
)
from optimodel_server_types.providerTypes import MakeQueryResponse

logger = logging.getLogger(__name__)


async def queryModel(
    model: ModelTypes,
    messages: list[ModelMessage],
    speedPriority: SpeedPriority = None,
    validator: Callable[[str], bool] = None,
    fallbackModels: list[ModelTypes] = [],
    maxGenLen: int = None,
    temperature: float = 0.2,
    jsonMode: bool = None,
    provider: Providers | None = None,
    userId: str | None = None,
    sessionId: str | None = None,
    guards: list[Guards] | None = None,
    retries: int | None = None,
    timeout: int | None = None,
    workflowName: str | None = None,
    credentials: list[Credentials] | None = None,
) -> MakeQueryResponse:
    """
    Query a model
    @param model: The model to use
    @param messages: The messages to send to the model
    @param speedPriority: The speed priority of the query
    @param validator: A function that takes in the model output and returns a boolean if it passed/failed validation
    @param fallbackModels: A list of models to use if the first model fails.
    @param jsonMode: Whether to return the response in JSON mode
    @param userId: [Lytix Specific] The user id to use for the query
    @param sessionId: [Lytix Specific] The session id to use for the query
    @param guard: A list of guards to use for the query
    @param retries: The number of retries to attempt if the model fails.
    @param timeout: The timeout in seconds to wait for the model to respond.
    @param workflowName: [Lytix Specific] The workflow name to use for the query
    """
    # Either 0 retries or whatevers passed
    retriesParsed = retries if retries else 1

    while retriesParsed > 0:
        # Make our request
        try:
            async with aiohttp.ClientSession(
                json_serialize=lambda object: json.dumps(
                    object, indent=4, cls=ObjectEncoder
                ),
                timeout=aiohttp.ClientTimeout(total=timeout) if timeout else None,
            ) as session:
                try:
                    allModels = [model, *fallbackModels]
                    for index, model in enumerate(allModels):
                        try:
                            """
                            Convert model int to string
                            """
                            modelToUse = ModelTypes(model).name
                            body = {
                                "modelToUse": modelToUse,
                                "messages": messages,
                                "speedPriority": speedPriority,
                                "temperature": temperature,
                                "jsonMode": jsonMode,
                                "provider": provider.name if provider else None,
                                "userId": userId if userId else None,
                                "sessionId": sessionId if sessionId else None,
                                "guards": guards,
                                "workflowName": workflowName if workflowName else None,
                                "credentials": credentials,
                            }
                            if maxGenLen:
                                body["maxGenLen"] = maxGenLen
                            async with session.post(
                                url=f"{LytixCreds.LX_BASE_URL.rstrip('/')}/optimodel/api/v1/query",
                                json=body,
                                headers={
                                    "Authorization": f"Bearer {LytixCreds.LX_API_KEY}",
                                    "Content-Type": "application/json",
                                },
                            ) as response:
                                jsonResponse = await response.json()
                                if jsonResponse.get("modelResponse", None) is None:
                                    raise Exception(f"Bad request: {jsonResponse}")
                                if validator:
                                    if not validator(jsonResponse["modelResponse"]):
                                        logger.warn(
                                            f"Failed validation when trying model {model}"
                                        )
                                        raise Exception("Validation failed")

                                return jsonResponse
                        except Exception as e:
                            """
                            If we have any more models to try, try them first
                            """
                            # If we got more models to try
                            if index < len(allModels) - 1:
                                continue
                            else:
                                raise e
                except Exception as e:
                    raise e
        except Exception as e:
            retriesParsed -= 1

            # If we have more retries to try
            if retriesParsed > 0 and retries and retries > 0:
                logger.warn(
                    f"Retrying model {model} due to error: {e}. Remaining retries: {retriesParsed}"
                )
                try:
                    # Do exponential backoff
                    sleepTime = 2 ** ((retries - retriesParsed) + 3)
                    logger.info(f"Sleeping for {sleepTime} seconds")
                    await asyncio.sleep(sleepTime)
                except Exception as e:
                    logger.warn(f"Error sleeping. Going to do it normally. Error: {e}")
                    await asyncio.sleep(15)
            else:
                raise e
    raise Exception("Failed to query model")


class ObjectEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__
