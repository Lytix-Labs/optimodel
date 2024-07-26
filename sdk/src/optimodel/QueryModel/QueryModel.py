import json
import logging
from typing import Callable
import aiohttp

from optimodel.envVars import LytixCreds
from optimodel_server_types import (
    ModelImageMessageSource,
    ModelMessage,
    ModelMessageContentEntry,
    SpeedPriority,
    ModelTypes,
    Providers,
)


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
    workflowId: str | None = None,
):
    """
    Query a model
    @param model: The model to use
    @param messages: The messages to send to the model
    @param speedPriority: The speed priority of the query
    @param validator: A function that takes in the model output and returns a boolean if it passed/failed validation
    @param fallbackModels: A list of models to use if the first model fails.
    @param jsonMode: Whether to return the response in JSON mode
    @param userId: The user id to use for the query
    @param workflowId: The workflow id to use for the query
    """
    async with aiohttp.ClientSession(
        json_serialize=lambda object: json.dumps(object, indent=4, cls=ObjectEncoder)
    ) as session:
        try:
            allModels = [model, *fallbackModels]
            for index, model in enumerate(allModels):
                try:
                    """
                    Convert model int to string
                    """
                    modelToUse = ModelTypes(model).name
                    async with session.post(
                        url=f"{LytixCreds.LX_BASE_URL.rstrip('/')}/optimodel/api/v1/query",
                        json={
                            "modelToUse": modelToUse,
                            "messages": messages,
                            "speedPriority": speedPriority,
                            "maxGenLen": maxGenLen,
                            "temperature": temperature,
                            "jsonMode": jsonMode,
                            "provider": provider.name if provider else None,
                            "userId": userId if userId else None,
                            "workflowId": workflowId if workflowId else None,
                        },
                        headers={"Authorization": f"Bearer {LytixCreds.LX_API_KEY}"},
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

                        return jsonResponse["modelResponse"]
                except Exception as e:
                    """
                    If we have any more models to try, try them first
                    """
                    if index < len(allModels) - 1:
                        continue
                    else:
                        raise e
        except Exception as e:
            raise e


class ObjectEncoder(json.JSONEncoder):
    def default(self, o):
        if (
            isinstance(o, ModelMessage)
            or isinstance(o, ModelMessageContentEntry)
            or isinstance(o, ModelImageMessageSource)
        ):
            return o.__dict__
        return o
