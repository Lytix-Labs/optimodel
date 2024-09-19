import os

from groq import NOT_GIVEN, Groq
from optimodel_server.OptimodelError import OptimodelError
from optimodel_server.Providers.CommonUtils import containsImageInMessages


from optimodel_types import (
    GroqCredentials,
    ModelTypes,
)
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
    QueryParams,
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
        params: QueryParams,
    ):
        messages = params["messages"]
        model = params["model"]
        temperature = params.get("temperature", None)
        maxGenLen = params.get("maxGenLen", None)
        credentials = params.get("credentials", None)
        jsonMode = params.get("jsonMode", False)

        if jsonMode is True:
            raise OptimodelError("JSON mode not supported for Groq")

        """
        @NOTE Groq does not currently support image types
        """
        if containsImageInMessages(messages):
            raise OptimodelError(
                "Groq does not currently support image types", provider="groq"
            )

        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Groq credentials not provided", provider="groq")

            # Try to find the together credentials
            groqCreds = next(
                (x for x in credentials if type(x) == GroqCredentials), None
            )
            if groqCreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Groq credentials not found", provider="groq")

            client = Groq(api_key=groqCreds.groqApiKey)
        else:
            if self.groqClient is None:
                raise OptimodelError("Groq client not initialized", provider="groq")
            client: Groq = self.groqClient

        match model:
            case ModelTypes.llama_3_8b_instruct.name:
                modelId = "llama3-8b-8192"
            case ModelTypes.llama_3_70b_instruct.name:
                modelId = "llama3-70b-8192"
            case ModelTypes.mixtral_8x7b_instruct.name:
                modelId = "mixtral-8x7b-32768"
            case ModelTypes.llama_3_1_405b.name:
                modelId = "llama-3.1-405b-reasoning"
            case ModelTypes.llama_3_1_70b.name:
                modelId = "llama-3.1-70b-versatile"
            case ModelTypes.llama_3_1_8b.name:
                modelId = "llama-3.1-8b-instant"
            case _:
                raise OptimodelError(f"Model {model} not supported", provider="groq")

        messagesParsed = []
        for message in messages:
            if isinstance(message.content, str):
                messagesParsed.append(
                    {"role": message.role, "content": message.content}
                )
            else:
                # Get the text from message.content
                textForRole = next(
                    (x.text for x in message.content if x.type == "text"), None
                )
                if textForRole is not None:
                    messagesParsed.append(
                        {"role": message.role, "content": textForRole}
                    )
                else:
                    raise OptimodelError(
                        f"No text found for role {message.role}", provider="groq"
                    )

        response = client.chat.completions.create(
            model=modelId,
            messages=messagesParsed,
            temperature=temperature,
            max_tokens=maxGenLen if maxGenLen else NOT_GIVEN,
        )

        promptTokenCount = response.usage.prompt_tokens
        generationTokenCount = response.usage.completion_tokens
        modelOutput = response.choices[0].message.content
        return QueryResponse(
            modelOutput=modelOutput,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
