from typing import List
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from optimodel_server.GuardClient import GuardClient
from optimodel_server.OptimodelError import OptimodelError, OptimodelGuardError
from optimodel_server.Config import config
from optimodel_server.Planner import getAllAvailableProviders, orderProviders
from optimodel_server.Utils.QueryModelMain import queryModelMain
from optimodel_types import QueryBody
from optimodel_server.Config.types import SAAS_MODE
from optimodel_types.providerTypes import (
    MakeQueryResponse,
    QueryParams,
    QueryResponse,
    GuardError,
)

import logging
import sys
from fastapi import FastAPI
from optimodel_server.Routes.Proxy.AnthropicProxy import anthropicRouter
from optimodel_server.Routes.Proxy.OpenAIProxy import openaiRouter

from optimodel_server.Routes.Proxy.GeminiProxy import geminiRouter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Include the Anthropic proxy router
app.include_router(anthropicRouter, prefix="/proxy/v1/anthropic", tags=["proxy"])

# Include the OpenAI proxy router
app.include_router(openaiRouter, prefix="/proxy/v1/openai", tags=["proxy"])

# Include the Gemini proxy router
app.include_router(geminiRouter, prefix="/proxy/v1/gemini", tags=["proxy"])

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
    response = await queryModelMain(data, guardClientInstance)
    return response


@app.get(f"{baseURL}/list-models")
async def listModels():
    return {"models": config.modelToProvider}


@app.get(f"{baseURL}/health")
async def getHealth():
    return {"status": "ok"}
