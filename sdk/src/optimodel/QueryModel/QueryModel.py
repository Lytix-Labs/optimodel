import json
import logging
from typing import Callable
import aiohttp

from optimodel.Consts import BASE_URL
from optimodel.QueryModel.types import ModelTypes
from optimodel_server import ModelMessage, SpeedPriority


logger = logging.getLogger(__name__)


async def queryModel(
    model: ModelTypes,
    messages: list[ModelMessage],
    speedPriority: SpeedPriority = None,
    validator: Callable[[str], bool] = None,
    fallbackModels: list[ModelTypes] = [],
    maxGenLen: int = 1024,
    temperature: float = 0.2,
):
    """
    Query a model
    @param model: The model to use
    @param messages: The messages to send to the model
    @param speedPriority: The speed priority of the query
    @param validator: A function that takes in the model output and returns a boolean if it passed/failed validation
    @param fallbackModels: A list of models to use if the first model fails.
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
                        url=f"{BASE_URL.rstrip('/')}/query",
                        json={
                            "modelToUse": modelToUse,
                            "messages": messages,
                            "speedPriority": speedPriority,
                            "maxGenLen": maxGenLen,
                            "temperature": temperature,
                        },
                    ) as response:
                        jsonResponse = await response.json()
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
            print("Error querying model:", e)
            raise e


class ObjectEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ModelMessage):
            return o.__dict__
        return o
