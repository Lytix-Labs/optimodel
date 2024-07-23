import json
import os

from openai import OpenAI
from optimodel_server.OptimodelError import OptimodelError

from optimodel_server_types import ModelMessage, OpenAICredentials, ModelTypes
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
)


class OpenAIProvider(BaseProviderClass):
    supportSAASMode = True

    def __init__(self):
        if os.environ.get("OPEN_AI_KEY", None):
            self.openAIClient = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))

    def validateProvider(self):
        """
        Validate the accounts API works
        @TODO This should be a more robust check
        """
        if os.environ.get("OPEN_AI_KEY", None) is None:
            return False
        return True

    def makeQuery(
        self,
        messages: list[ModelMessage],
        model: ModelTypes,
        temperature: int = 0.2,
        maxGenLen: int = 1024,
        credentials: OpenAICredentials | None = None,
    ):
        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Together credentials not provided")

            # Try to find the together credentials
            openAICreds = next(
                (x for x in credentials if type(x) == OpenAICredentials), None
            )
            if openAICreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError("OpenAI credentials not found")

            client = OpenAI(api_key=openAICreds.openAiKey)
        else:
            if self.openAIClient is None:
                raise OptimodelError("OpenAI client not initialized")
            client = self.openAIClient

        match model:
            case ModelTypes.gpt_4.name:
                modelId = "gpt-4"
            case ModelTypes.gpt_3_5_turbo.name:
                modelId = "gpt-3.5-turbo"
            case ModelTypes.gpt_4o.name:
                modelId = "gpt-4o"
            case ModelTypes.gpt_4_turbo.name:
                modelId = "gpt-4-turbo"
            case ModelTypes.gpt_3_5_turbo_0125.name:
                modelId = "gpt-3.5-turbo-0125"
            case _:
                raise OptimodelError(f"Model {model} not supported")

        messageToPass = []
        for message in messages:
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
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{entry.source.mediaType};{entry.source.type},{entry.source.data}",
                                },
                            }
                        )

                messageToPass.append({"role": message.role, "content": baseContent})

        response = client.chat.completions.create(
            model=modelId,
            messages=messageToPass,
            temperature=temperature,
            max_tokens=maxGenLen,
        )
        promptTokenCount = response.usage.prompt_tokens
        generationTokenCount = response.usage.completion_tokens
        modelOutput = response.choices[0].message.content
        return QueryResponse(
            modelOutput=modelOutput,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
