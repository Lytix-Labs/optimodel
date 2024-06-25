from fastapi import FastAPI
from fastapi.responses import JSONResponse

from optimodel_server.Config import config
from optimodel_server.Planner import getAllAvailableProviders, orderProviders
from optimodel_server.RequestTypes import QueryBody
from optimodel_server.Config.types import SAAS_MODE

import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/query")
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
        for potentialProvider in orderedProviders:
            try:
                logger.info(f"Attempting query with {potentialProvider['provider']}...")

                # If we're in SAAS mode, validate we have credentials
                if SAAS_MODE is not None:
                    if data.credentials is None:
                        raise Exception("No credentials provided")

                response = config.providerInstances[
                    potentialProvider["provider"]
                ].makeQuery(
                    messages=data.messages,
                    model=potentialProvider["name"],
                    credentials=data.credentials,
                )

                if response:
                    logger.info(
                        f"Query successful with {potentialProvider['provider']}"
                    )
                    return {
                        "modelResponse": response.modelOutput,
                        "promptTokens": response.promptTokens,
                        "generationTokens": response.generationTokens,
                    }
            except Exception as e:
                logger.error(
                    f"Error with provider {potentialProvider['provider']}: {e}"
                )
                continue

        """
        Otherwise something bad has happened
        """
        raise Exception("No available provider")
    except Exception as e:
        logger.error(f"Error getting all available providers: {e}")
        return JSONResponse(
            status_code=500, content={"error": "Failed to retrieve providers"}
        )
