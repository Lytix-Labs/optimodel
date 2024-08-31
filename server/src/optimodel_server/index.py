from typing import List
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from optimodel_server.GuardClient import GuardClient
from optimodel_server.OptimodelError import OptimodelError, OptimodelGuardError
from optimodel_server.Config import config
from optimodel_server.Planner import getAllAvailableProviders, orderProviders
from optimodel_server_types import QueryBody
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server_types.providerTypes import (
    MakeQueryResponse,
    QueryParams,
    QueryResponse,
    GuardError,
)

import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

baseURL = "/optimodel/api/v1"

guardClientInstance = GuardClient(config.guardServerURL)


@app.on_event("startup")
async def startup_event():
    logger.info(f"🌐 Starting Optimodel Server...")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logger.error(request, exc_str[:1000])
    content = {"status_code": 10422, "message": "Invalid request", "data": None}
    return JSONResponse(content=content, status_code=422)


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
        Extract out any guards from the query if present
        """
        guards = data.guards
        logger.info(f"Guards: {guards}")

        """
        If we have any guards, split them up based on pre vs post query
        """
        preQueryGuards = []
        postQueryGuards = []
        if guards:
            for guard in guards:
                if guard.guardType == "preQuery":
                    preQueryGuards.append(guard)
                if guard.guardType == "postQuery":
                    postQueryGuards.append(guard)

        """
        Now attempt the query with each provider in order
        """
        errors = []
        guardErrors: List[GuardError] = []
        for potentialProvider in orderedProviders:
            try:
                providerName = potentialProvider["provider"]
                logger.info(
                    f"Attempting query model {data.modelToUse} with {providerName}..."
                )

                # If we're in SAAS mode, validate we have credentials
                if SAAS_MODE is not None:
                    if data.credentials is None:
                        raise OptimodelError("No credentials provided")

                """
                Check if we have any guards
                """
                if preQueryGuards:
                    for guard in preQueryGuards:
                        logger.info(f"Checking preQuery guard {guard.guardName}")
                        try:
                            guardResponse = await guardClientInstance.checkGuard(
                                guards=guard, messages=data.messages
                            )
                        except Exception as e:
                            logger.error(f"Error checking guard: {e}")
                            if guard.blockRequest is True:
                                raise OptimodelError(f"Error checking guard: {e}")
                            else:
                                guardResponse = {"failure": False}
                        print(f"guardResponse: {guardResponse}")
                        if guardResponse["failure"] is True:
                            guardErrors.append(
                                GuardError(
                                    guardName=guard.guardName,
                                    failure=True,
                                    metadata=guardResponse["metadata"],
                                    blockRequest=guard.blockRequest,
                                )
                            )
                            if guard.blockRequest is True:
                                # Short circuit calling the model
                                queryResponse: QueryResponse = {
                                    "modelResponse": (
                                        guard.blockRequestMessage
                                        if guard.blockRequestMessage
                                        else ""
                                    ),
                                    "promptTokens": 0,
                                    "generationTokens": 0,
                                    "cost": 0,
                                    "provider": providerName,
                                    "guardErrors": guardErrors,
                                }
                                return queryResponse

                maxGenLen = data.maxGenLen

                try:
                    params: QueryParams = {
                        "messages": data.messages,
                        "model": potentialProvider["name"],
                        "credentials": data.credentials,
                        "maxGenLen": maxGenLen,
                        "jsonMode": data.jsonMode,
                    }
                    response = config.providerInstances[providerName].makeQuery(
                        params=params
                    )
                except Exception as e:
                    logger.error(f"Error making query: {e}")
                    raise OptimodelError(
                        f"Error making query: {e}",
                        provider=providerName,
                    )

                if response:
                    logger.info(f"Query successful with {providerName}")
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
                    queryResponse: MakeQueryResponse = {
                        "modelResponse": response.modelOutput,
                        "promptTokens": response.promptTokens,
                        "generationTokens": response.generationTokens,
                        "cost": cost,
                        "provider": providerName,
                        "guardErrors": guardErrors,
                    }

                    """
                    Check if we have any guards
                    """
                    if postQueryGuards:
                        for guard in postQueryGuards:
                            logger.info(f"Checking postQuery guard {guard.guardName}")
                            guardResponse = await guardClientInstance.checkGuard(
                                guards=guard,
                                messages=data.messages,
                                modelOutput=response.modelOutput,
                            )
                            if guardResponse["failure"] is True:
                                guardErrors.append(
                                    GuardError(
                                        guardName=guard.guardName,
                                        failure=True,
                                        metadata=guardResponse["metadata"],
                                        blockRequest=guard.blockRequest,
                                    )
                                )
                                if guard.blockRequest is True:
                                    # If the user wants to return a custom message
                                    queryResponse["modelResponse"] = (
                                        guard.blockRequestMessage
                                        if guard.blockRequestMessage
                                        else ""
                                    )
                                    queryResponse["guardErrors"] = guardErrors

                    return queryResponse
            except Exception as e:
                logger.error(f"Error with provider {providerName}: {e}")
                if isinstance(e, OptimodelError):
                    # Add to list of errors
                    errors.append(e)
                if isinstance(e, OptimodelGuardError):
                    """
                    Nothing left to re-check, failed a guard, return that as an error
                    """
                    return JSONResponse(status_code=503, content={"error": e.guard})
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
