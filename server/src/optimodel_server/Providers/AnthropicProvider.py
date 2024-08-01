import os
from optimodel_server import OptimodelError

import anthropic

from optimodel_server_types import (
    AnthropicCredentials,
    ModelTypes,
)
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
    QueryParams,
)


class AnthropicProvider(BaseProviderClass):
    supportSAASMode = True
    supportJSONMode = False

    def __init__(self):
        if os.environ.get("ANTHROPIC_API_KEY", None):
            self.anthropicClient = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )

    def validateProvider(self):
        """
        Validate the accounts API works
        @TODO This should be a more robust check
        """
        if os.environ.get("ANTHROPIC_API_KEY", None) is None:
            return False
        return True

    def makeQuery(
        self,
        params: QueryParams,
    ):
        messages = params["messages"]
        model = params["model"]
        temperature = params.get("temperature", None)
        maxGenLen = params.get("maxGenLen", None)
        credentials = params.get("credentials", None)
        jsonMode = params.get("jsonMode", False)

        if jsonMode is not None:
            raise OptimodelError("JSON mode not supported for Anthropic")

        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Anthropic credentials not provided")

            # Try to find the anthropic credentials
            anthropicCreds = next(
                (x for x in credentials if type(x) == AnthropicCredentials), None
            )
            if anthropicCreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Anthropic credentials not found")

            client = anthropic.Anthropic(api_key=anthropicCreds.anthropicApiKey)
        else:
            if self.anthropicClient is None:
                raise OptimodelError("Anthropic client not initialized")
            client = self.anthropicClient

        match model:
            case ModelTypes.claude_3_5_sonnet.name:
                modelId = "claude-3-sonnet-20240229"
            case ModelTypes.claude_3_haiku.name:
                modelId = "claude-3-haiku-20240307"
            case _:
                raise OptimodelError(
                    f"Model {model} not supported", provider="anthropic"
                )

        messageToPass = []
        systemMessage = None
        for message in messages:
            if message.role == "system":
                systemMessage = message
                continue

            if isinstance(message.content, str):
                messageToPass.append({"role": message.role, "content": message.content})
            else:
                baseContent = []
                """
                Loop over the content
                """
                for entry in message.content:
                    if isinstance(entry, str):
                        baseContent.append({"type": "text", "text": entry})
                    elif entry.type == "text":
                        baseContent.append(
                            {
                                "type": "text",
                                "text": entry.text,
                            }
                        )
                    elif entry.type == "image":
                        baseContent.append(
                            {
                                "type": "image",
                                "source": {
                                    "data": entry.source.data,
                                    "media_type": entry.source.mediaType,
                                    "type": entry.source.type,
                                },
                            }
                        )
                messageToPass.append({"role": message.role, "content": baseContent})

        response = client.messages.create(
            model=modelId,
            system=systemMessage.content if systemMessage else None,
            messages=messageToPass,
            temperature=temperature,
            max_tokens=maxGenLen if maxGenLen else anthropic.NOT_GIVEN,
        )
        promptTokenCount = response.usage.input_tokens
        generationTokenCount = response.usage.output_tokens
        modelOutput = response.content[0].text
        return QueryResponse(
            modelOutput=modelOutput,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
