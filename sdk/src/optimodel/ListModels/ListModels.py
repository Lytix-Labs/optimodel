import aiohttp

from optimodel.envVars import LytixCreds
from optimodel.Utils import HttpClient


async def listModels():
    """
    List all available models
    """
    response = await HttpClient.getRequest(
        f"{LytixCreds.LX_BASE_URL.rstrip('/')}/optimodel/api/v1/list-models"
    )
    return response
