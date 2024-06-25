import aiohttp

from optimodel.Consts import BASE_URL
from optimodel.Utils import HttpClient


async def listModels():
    """
    List all available models
    """
    response = await HttpClient.getRequest(f"{BASE_URL.rstrip('/')}/list-models")
    return response
