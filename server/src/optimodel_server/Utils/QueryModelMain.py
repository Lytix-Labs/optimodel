import json
from typing import List, Tuple, Dict, Any
from fastapi.responses import JSONResponse
from httpx import QueryParams
from optimodel_server import GuardClient
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.OptimodelError import OptimodelError, OptimodelGuardError
from optimodel_server.Planner.Planner import getAllAvailableProviders, orderProviders
from optimodel_types import Guards, ModelMessage, QueryBody
from optimodel_types.providerTypes import GuardError, MakeQueryResponse, QueryResponse
import logging
import sys
from optimodel_server.Config import config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_pre_query_guards(
    preQueryGuards: List[Any],
    guardClientInstance: GuardClient,
    messages: List[ModelMessage],
    providerName: str,
) -> Tuple[List[GuardError], bool]:
    guardErrors = []
    for guard in preQueryGuards:
        logger.info(f"Checking preQuery guard {guard.guardName}")
        try:
            guardResponse = await guardClientInstance.checkGuard(
                guards=guard, messages=messages
            )
        except Exception as e:
            logger.error(f"Error checking guard: {e}")
            if guard.blockRequest is True:
                raise OptimodelError(f"Error checking guard: {e}")
            else:
                guardResponse = {"failure": False}

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
                return {
                    "modelResponse": (
                        guard.blockRequestMessage if guard.blockRequestMessage else ""
                    ),
                    "promptTokens": 0,
                    "generationTokens": 0,
                    "cost": 0,
                    "provider": providerName,
                    "guardErrors": guardErrors,
                }, True
    return guardErrors, False


async def queryModelMain(data: QueryBody, guardClientInstance: GuardClient):
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
        finalGuardErrors: List[GuardError] = []
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
                    preQueryGuardErrors, should_return = await check_pre_query_guards(
                        preQueryGuards=preQueryGuards,
                        guardClientInstance=guardClientInstance,
                        messages=data.messages,
                        providerName=providerName,
                    )
                    if should_return:
                        """
                            Weird edge case where we want to return the guard error
                        @TODO: This is horrible code, refactor it
                        """
                        return preQueryGuardErrors

                    if preQueryGuardErrors:
                        finalGuardErrors.extend(preQueryGuardErrors)

                maxGenLen = data.maxGenLen

                try:
                    params: QueryParams = {
                        "messages": data.messages,
                        "model": potentialProvider["name"],
                        "credentials": data.credentials,
                        "maxGenLen": maxGenLen,
                        "jsonMode": data.jsonMode,
                        "temperature": data.temperature,
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
                        if response.promptTokens > 128_000 and potentialProvider.get(
                            "pricePer1MInputAbove125K"
                        ):
                            inputCost = response.promptTokens * (
                                potentialProvider["pricePer1MInputAbove125K"]
                                / 1_000_000
                            )
                        else:
                            inputCost = response.promptTokens * (
                                potentialProvider["pricePer1MInput"] / 1_000_000
                            )
                        if (
                            response.generationTokens > 128_000
                            and potentialProvider.get("pricePer1MOutputAbove125K")
                        ):
                            outputCost = response.generationTokens * (
                                potentialProvider["pricePer1MOutputAbove125K"]
                                / 1_000_000
                            )
                        else:
                            outputCost = response.generationTokens * (
                                potentialProvider["pricePer1MOutput"] / 1_000_000
                            )
                        cost = inputCost + outputCost
                        logger.info(
                            f">>>Cost: {cost}, {json.dumps(potentialProvider, indent=2)}"
                        )
                    except Exception as e:
                        logger.error(f"Error getting cost: {e}")
                        cost = None
                    queryResponse: MakeQueryResponse = {
                        "modelResponse": response.modelOutput,
                        "promptTokens": response.promptTokens,
                        "generationTokens": response.generationTokens,
                        "cost": cost,
                        "provider": providerName,
                        "guardErrors": finalGuardErrors,
                    }

                    """
                    Check if we have any guards
                    """
                    if postQueryGuards:
                        postQueryGuardErrors, queryResponse = (
                            await check_post_query_guards(
                                postQueryGuards=postQueryGuards,
                                guardClientInstance=guardClientInstance,
                                messages=data.messages,
                                modelOutput=response.modelOutput,
                                queryResponse=queryResponse,
                            )
                        )
                        if postQueryGuardErrors:
                            queryResponse["guardErrors"].extend(postQueryGuardErrors)

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


async def check_post_query_guards(
    postQueryGuards: List[Guards],
    guardClientInstance: GuardClient,
    messages: List[ModelMessage],
    modelOutput: str,
    queryResponse: MakeQueryResponse,
) -> Tuple[List[GuardError], MakeQueryResponse]:
    guardErrors = []
    for guard in postQueryGuards:
        logger.info(f"Checking postQuery guard {guard.guardName}")
        guardResponse = await guardClientInstance.checkGuard(
            guards=guard,
            messages=messages,
            modelOutput=modelOutput,
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
                    guard.blockRequestMessage if guard.blockRequestMessage else ""
                )
                queryResponse["guardErrors"] = guardErrors
                return guardErrors, queryResponse
    return guardErrors, queryResponse
