import json
import os

from together import Together

from optimodel_server.RequestTypes import ModelMessage
from optimodel_server.Config.types import ModelTypes
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
)


class TogetherProvider(BaseProviderClass):
    def __init__(self):
        if os.environ.get("TOGETHER_API_KEY", None):
            self.togetherClient = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    def validateProvider(self):
        """
        Validate the accounts API works
        @TODO This should be a more robust check
        """
        if os.environ.get("TOGETHER_API_KEY", None) is None:
            return False
        return True

    def makeQuery(
        self,
        messages: list[ModelMessage],
        model: ModelTypes,
        temperature: int = 0.2,
        maxGenLen: int = 1024,
    ):
        match model:
            case ModelTypes.llama3_8b_instruct.name:
                modelId = "meta-llama/Llama-3-8b-chat-hf"
            case ModelTypes.llama3_70b_instruct.name:
                modelId = "meta-llama/Llama-3-70b-chat-hf"
            case _:
                raise Exception(f"Model {model} not supported")

        response = self.togetherClient.chat.completions.create(
            model=modelId,
            messages=[{"role": x.role, "content": x.content} for x in messages],
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
