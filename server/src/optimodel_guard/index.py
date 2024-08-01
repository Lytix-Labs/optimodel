from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from optimodel_guard.Guards import GuardBaseClass, GuardMapping

from optimodel_server_types import GuardBody, GuardResponse

import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

baseURL = "/optimodel-guard/api/v1"


@app.on_event("startup")
async def startup_event():
    logger.info(f"🔐 Starting Optimodel Guard Server...")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logger.error(request, exc_str)
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=422)


@app.post(f"{baseURL}/guard")
async def query(data: GuardBody):
    logger.info(f"New request to check guards...")

    """
    Get an instance of the guard passed
    """
    guard: GuardBaseClass = GuardMapping[data.guard.guardName]

    """
    Then check the guard
    """
    if data.guard.guardType == "preQuery":
        guardResponse = guard.handlePreQuery(data.messages)
    elif data.guard.guardType == "postQuery":
        guardResponse = guard.handlePostQuery(data.messages, data.response)

    toReturn = GuardResponse(failure=guardResponse)
    return toReturn


@app.get(f"{baseURL}/health")
async def getHealth():
    return {"status": "ok"}
