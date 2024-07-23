import json
import os
from optimodel_server import OptimodelError

from together import Together

from optimodel_server.Providers.CommonUtils import containsImageInMessages
from optimodel_server_types import ModelMessage, TogetherAICredentials, ModelTypes
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
)


class TogetherProvider(BaseProviderClass):
    supportSAASMode = True

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
        credentials: TogetherAICredentials | None = None,
    ):
        """
        @NOTE Together does not currently support image types
        """
        if containsImageInMessages(messages):
            raise OptimodelError("Together does not currently support image types")

        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Together credentials not provided")

            # Try to find the together credentials
            togetherCreds = next(
                (x for x in credentials if type(x) == TogetherAICredentials), None
            )
            if togetherCreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Together credentials not found")

            client = Together(api_key=togetherCreds.togetherApiKey)
        else:
            if self.togetherClient is None:
                raise OptimodelError("Together client not initialized")
            client = self.togetherClient

        match model:
            case ModelTypes.llama_3_8b_instruct.name:
                modelId = "meta-llama/Llama-3-8b-chat-hf"
            case ModelTypes.llama_3_70b_instruct.name:
                modelId = "meta-llama/Llama-3-70b-chat-hf"
            case _:
                raise OptimodelError(f"Model {model} not supported")

        response = client.chat.completions.create(
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
