import os
from optimodel_server.OptimodelError import OptimodelError
from mistralai import Mistral

from optimodel_server.Providers.CommonUtils import containsImageInMessages
from optimodel_types import (
    MistralAICredentials,
    ModelTypes,
)
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
    QueryParams,
)


class MistralAIProvider(BaseProviderClass):
    supportSAASMode = True
    supportJSONMode = False

    def __init__(self):
        if os.environ.get("MISTRAL_API_KEY", None):
            self.mistralClient = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

    def validateProvider(self):
        """
        Validate the accounts API works
        @TODO This should be a more robust check
        """
        if os.environ.get("MISTRAL_API_KEY", None) is None:
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

        if jsonMode is True:
            raise OptimodelError("JSON mode not supported for MistralAI")

        """
        @NOTE Together does not currently support image types
        """
        if containsImageInMessages(messages):
            raise OptimodelError(
                "MistralAI does not currently support image types", provider="mistralai"
            )

        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError(
                    "MistralAI credentials not provided", provider="mistralai"
                )

            # Try to find the together credentials
            mistralCreds = next(
                (x for x in credentials if type(x) == MistralAICredentials), None
            )
            if mistralCreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError(
                    "MistralAI credentials not found", provider="mistralai"
                )

            client = Mistral(api_key=mistralCreds.mistralApiKey)
        else:
            if self.mistralClient is None:
                raise OptimodelError(
                    "MistralAI client not initialized", provider="mistralai"
                )
            client = self.mistralClient

        match model:
            case ModelTypes.open_mistral_nemo.name:
                modelId = "open-mistral-nemo"
            case ModelTypes.mistral_large_latest.name:
                modelId = "mistral-large-latest"
            case ModelTypes.codestral_latest.name:
                modelId = "codestral-latest"
            case _:
                raise OptimodelError(
                    f"Model {model} not supported", provider="mistralai"
                )

        finalMessages = []
        for message in messages:
            if isinstance(message.content, str):
                finalMessages.append({"role": message.role, "content": message.content})
            else:
                # Get the text from message.content
                textForRole = next(
                    (x.text for x in message.content if x.type == "text"), None
                )
                if textForRole is not None:
                    finalMessages.append({"role": message.role, "content": textForRole})
                else:
                    raise OptimodelError(
                        f"No text found for role {message.role}", provider="groq"
                    )

        response = client.chat.complete(
            model=modelId,
            messages=finalMessages,
            temperature=temperature,
            max_tokens=maxGenLen if maxGenLen else None,
        )
        promptTokenCount = response.usage.prompt_tokens
        generationTokenCount = response.usage.completion_tokens
        modelOutput = response.choices[0].message.content
        return QueryResponse(
            modelOutput=modelOutput,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
