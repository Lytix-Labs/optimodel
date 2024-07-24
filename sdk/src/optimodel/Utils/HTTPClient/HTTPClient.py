import aiohttp

from optimodel.envVars import LytixCreds


class HttpClient:
    @staticmethod
    async def getRequest(url: str) -> aiohttp.ClientResponse:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url=url,
                headers={"Authorization": f"Bearer {LytixCreds.LX_API_KEY}"},
            ) as response:
                return await response.json()
