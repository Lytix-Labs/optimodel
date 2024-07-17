import json
import os

from groq import Groq


from optimodel_server_types import (
    GroqCredentials,
    ModelMessage,
    ModelTypes,
)
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
)


class GroqProvider(BaseProviderClass):
    supportSAASMode = True
    groqClient: Groq

    def __init__(self):
        if os.environ.get("GROQ_API_KEY", None):
            self.groqClient = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def validateProvider(self):
        """
        Validate the accounts API works
        @TODO This should be a more robust check
        """
        if os.environ.get("GROQ_API_KEY", None) is None:
            return False
        return True

    def makeQuery(
        self,
        messages: list[ModelMessage],
        model: ModelTypes,
        temperature: int = 0.2,
        maxGenLen: int = 1024,
        credentials: GroqCredentials | None = None,
    ):
        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise Exception("Together credentials not provided")

            # Try to find the together credentials
            groqCreds = next(
                (x for x in credentials if type(x) == GroqCredentials), None
            )
            if groqCreds is None:
                # This should have been filtered out in the planner
                raise Exception("Together credentials not found")

            client = Groq(api_key=groqCreds.groqApiKey)
        else:
            if self.groqClient is None:
                raise Exception("Groq client not initialized")
            client: Groq = self.groqClient

        match model:
            case ModelTypes.llama_3_8b_instruct.name:
                modelId = "llama3-8b-8192"
            case ModelTypes.llama_3_70b_instruct.name:
                modelId = "llama3-70b-8192"
            case ModelTypes.mixtral_8x7b_instruct.name:
                modelId = "mixtral-8x7b-32768"
            case _:
                raise Exception(f"Model {model} not supported")

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