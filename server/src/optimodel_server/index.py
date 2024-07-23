from fastapi import FastAPI
from fastapi.responses import JSONResponse
from optimodel_server.OptimodelError import OptimodelError

from optimodel_server.Config import config
from optimodel_server.Planner import getAllAvailableProviders, orderProviders
from optimodel_server_types import QueryBody
from optimodel_server.Config.types import SAAS_MODE

import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

baseURL = "/optimodel/api/v1"


@app.post(f"{baseURL}/query")
async def read_root(data: QueryBody):
    """
    Now its time to decide what model to use, first lets get a list of available
    providers
    """
    try:
        allAvailableProviders = getAllAvailableProviders(data)

        """
        Now its time to just pick the best one based on our criteria
        """
        orderedProviders = orderProviders(allAvailableProviders, data)

        """
        Now attempt the query with each provider in order
        """
        errors = []
        for potentialProvider in orderedProviders:
            try:
                logger.info(f"Attempting query with {potentialProvider['provider']}...")

                # If we're in SAAS mode, validate we have credentials
                if SAAS_MODE is not None:
                    if data.credentials is None:
                        raise OptimodelError("No credentials provided")

                if data.maxGenLen is None:
                    """
                    Use max for this provider
                    """
                    data.maxGenLen = potentialProvider["maxGenLen"]

                try:
                    response = config.providerInstances[
                        potentialProvider["provider"]
                    ].makeQuery(
                        messages=data.messages,
                        model=potentialProvider["name"],
                        credentials=data.credentials,
                        maxGenLen=data.maxGenLen,
                    )
                except Exception as e:
                    logger.error(f"Error making query: {e}")
                    raise OptimodelError(
                        f"Error making query: {e}",
                        provider=potentialProvider["provider"],
                    )

                if response:
                    logger.info(
                        f"Query successful with {potentialProvider['provider']}"
                    )
                    try:
                        inputCost = (
                            response.promptTokens
                            * potentialProvider["costPerTokenInput"]
                        )
                        outputCost = (
                            response.generationTokens
                            * potentialProvider["costPerTokenOutput"]
                        )
                        cost = inputCost + outputCost
                    except Exception as e:
                        logger.error(f"Error getting cost: {e}")
                        cost = None
                    return {
                        "modelResponse": response.modelOutput,
                        "promptTokens": response.promptTokens,
                        "generationTokens": response.generationTokens,
                        "cost": cost,
                        "provider": potentialProvider["provider"],
                    }
            except Exception as e:
                logger.error(
                    f"Error with provider {potentialProvider['provider']}: {e}"
                )
                if isinstance(e, OptimodelError):
                    # Add to list of errors
                    errors.append(e)
                continue

        # Otherwise something bad has happened
        error_messages = [str(e) for e in errors]
        raise OptimodelError(f"No available provider. Got errors: {error_messages}")
    except Exception as e:
        logger.error(f"Error getting all available providers: {e}")
        if isinstance(e, OptimodelError):
            return JSONResponse(status_code=503, content={"error": str(e)})
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "error": f"Unhandled error. Contact support@lytix.co for help."
                },
            )


@app.get(f"{baseURL}/list-models")
async def listModels():
    return {"models": config.modelToProvider}


@app.get(f"{baseURL}/health")
async def getHealth():
    return {"status": "ok"}
