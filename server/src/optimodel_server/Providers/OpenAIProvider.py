import os

from openai import NOT_GIVEN, OpenAI
from optimodel_server.OptimodelError import OptimodelError

from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
    QueryParams,
)
from optimodel_types import ModelTypes, OpenAICredentials


class OpenAIProvider(BaseProviderClass):
    supportSAASMode = True
    supportJSONMode = True

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
        params: QueryParams,
    ):

        messages = params["messages"]
        model = params["model"]
        temperature = params.get("temperature", None)
        maxGenLen = params.get("maxGenLen", None)
        credentials = params.get("credentials", None)
        jsonMode = params.get("jsonMode", False)

        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError(
                    "OpenAI credentials not provided", provider="openai"
                )

            # Try to find the together credentials
            openAICreds = next(
                (x for x in credentials if type(x) == OpenAICredentials), None
            )
            if openAICreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError("OpenAI credentials not found", provider="openai")

            client = OpenAI(api_key=openAICreds.openAiKey)
        else:
            if self.openAIClient is None:
                raise OptimodelError("OpenAI client not initialized", provider="openai")
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
            case ModelTypes.gpt_4o_mini.name:
                modelId = "gpt-4o-mini"
            case ModelTypes.gpt_4o_mini_2024_07_18.name:
                modelId = "gpt-4o-mini-2024-07-18"
            case ModelTypes.gpt_4o_2024_08_06.name:
                modelId = "gpt-4o-2024-08-06"
            case ModelTypes.gpt_4o_2024_05_13.name:
                modelId = "gpt-4o-2024-05-13"
            case ModelTypes.o1_preview.name:
                modelId = "o1-preview"
            case ModelTypes.o1_preview_2024_09_12.name:
                modelId = "o1-preview-2024-09-12"
            case ModelTypes.o1_mini.name:
                modelId = "o1-mini"
            case ModelTypes.o1_mini_2024_09_12.name:
                modelId = "o1-mini-2024-09-12"
            case _:
                raise OptimodelError(f"Model {model} not supported", provider="openai")

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

        if modelId.startswith("o1"):
            response = client.chat.completions.create(
                model=modelId,
                messages=messageToPass,
                **({"temperature": temperature} if temperature is not None else {}),
                max_completion_tokens=maxGenLen if maxGenLen else NOT_GIVEN,
                response_format={"type": "json_object"} if jsonMode else None,
            )
        else:
            response = client.chat.completions.create(
                model=modelId,
                messages=messageToPass,
                **({"temperature": temperature} if temperature is not None else {}),
                max_tokens=maxGenLen if maxGenLen else NOT_GIVEN,
                response_format={"type": "json_object"} if jsonMode else None,
            )
        promptTokenCount = response.usage.prompt_tokens
        generationTokenCount = response.usage.completion_tokens
        modelOutput = response.choices[0].message.content
        return QueryResponse(
            modelOutput=modelOutput,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
