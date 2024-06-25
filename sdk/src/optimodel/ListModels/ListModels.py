import aiohttp

from optimodel.Consts import BASE_URL


async def listModels():
    """
    List all available models
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url=f"{BASE_URL.rstrip('/')}/list-models",
        ) as response:
            jsonResponse = await response.json()
            return jsonResponse
