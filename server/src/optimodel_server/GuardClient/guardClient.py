import aiohttp
from urllib.parse import urljoin
import json

from optimodel_server_types import GuardResponse, ModelMessage, Guards


class GuardClient:
    def __init__(self, guardServerURL: str):
        self.guardServerURL = urljoin(guardServerURL, "optimodel-guard/api/v1/guard")

    async def checkGuard(
        self,
        guards: Guards,
        messages: list[ModelMessage],
        modelOutput: str | None = None,
    ) -> GuardResponse:
        """
        Check a single guard given an input
        """

        # Make our request to our guard server
        async with aiohttp.ClientSession(
            json_serialize=lambda object: json.dumps(
                object, indent=4, cls=GuardObjectEncoder
            )
        ) as session:
            if modelOutput:
                # Add it to messages with the assistant role
                messages.append(ModelMessage(role="assistant", content=modelOutput))
            async with session.post(
                self.guardServerURL,
                json={
                    "guard": guards,
                    "messages": messages,
                },
            ) as response:
                response_data = await response.json()
                return response_data


class GuardObjectEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__
