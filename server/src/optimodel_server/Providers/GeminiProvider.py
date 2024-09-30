import os

from optimodel_server.OptimodelError import OptimodelError

from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
    QueryParams,
)
from optimodel_types import GeminiCredentials, ModelTypes, OpenAICredentials
from google.ai import generativelanguage as glm
import google.generativeai as genai
from google.generativeai import protos


class GeminiProvider(BaseProviderClass):
    supportSAASMode = True
    supportJSONMode = True

    def __init__(self):
        if os.environ.get("GEMINI_API_KEY", None):
            self.geminiClient = glm.GenerativeServiceClient(
                client_options={"api_key": os.environ.get("GEMINI_API_KEY")}
            )

    def validateProvider(self):
        """
        Validate the accounts API works
        @TODO This should be a more robust check
        """
        if os.environ.get("GEMINI_API_KEY", None) is None:
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
                    "Gemini credentials not provided", provider="gemini"
                )

            # Try to find the together credentials
            geminiCreds = next(
                (x for x in credentials if type(x) == GeminiCredentials), None
            )
            if geminiCreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError("Gemini credentials not found", provider="gemini")

            client = glm.GenerativeServiceClient(
                client_options={"api_key": geminiCreds.geminiApiKey}
            )
        else:
            if self.geminiClient is None:
                raise OptimodelError("Gemini client not initialized", provider="gemini")
            client = self.geminiClient

        match model:
            case ModelTypes.gemini_1_5_pro.name:
                modelId = "models/gemini-1.5-pro"
            case ModelTypes.gemini_1_5_pro_001.name:
                modelId = "models/gemini-1.5-pro-001"
            case ModelTypes.gemini_1_5_pro_exp_0801.name:
                modelId = "models/gemini-1.5-pro-exp-0801"
            case ModelTypes.gemini_1_5_pro_exp_0827.name:
                modelId = "models/gemini-1.5-pro-exp-0827"
            case ModelTypes.gemini_1_5_flash.name:
                modelId = "models/gemini-1.5-flash"
            case ModelTypes.gemini_1_5_flash_latest.name:
                modelId = "models/gemini-1.5-flash-latest"
            case ModelTypes.gemini_1_5_flash_001.name:
                modelId = "models/gemini-1.5-flash-001"
            case ModelTypes.gemini_1_5_flash_001_tuning.name:
                modelId = "models/gemini-1.5-flash-001-tuning"
            case ModelTypes.gemini_1_5_flash_exp_0827.name:
                modelId = "models/gemini-1.5-flash-exp-0827"
            case ModelTypes.gemini_1_5_flash_8b_exp_0827.name:
                modelId = "models/gemini-1.5-flash-8b-exp-0827"
            case _:
                raise OptimodelError(f"Model {model} not supported", provider="gemini")

        """
        Attempt to extract the system instruction from the messages
        """
        systemInstruction = None
        for message in messages:
            if message.role == "system":
                if isinstance(message.content, str):
                    systemInstruction = message.content
                else:
                    systemInstruction = message.content[0].text
                break

        model = genai.GenerativeModel(
            modelId,
            system_instruction=systemInstruction,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": maxGenLen,
            },
        )
        model._client = client

        inputMessage = ""
        fileInput = None

        for message in messages:
            if message.role == "user":
                if isinstance(message.content, str):
                    inputMessage += message.content
                else:
                    for entry in message.content:
                        if isinstance(entry, str):
                            inputMessage += entry
                        elif entry.type == "text":
                            inputMessage += entry.text
                        elif entry.type == "video-gemini":
                            file_data = protos.FileData(
                                mime_type=entry.data.mimeType,
                                file_uri=entry.data.fileUri,
                            )
                            fileInput = file_data

        content = [inputMessage]
        if fileInput is not None:
            content.append(fileInput)

        response = model.generate_content(content)
        promptTokenCount = response.usage_metadata.prompt_token_count
        generationTokenCount = response.usage_metadata.candidates_token_count
        modelOutput = response.text
        return QueryResponse(
            modelOutput=modelOutput,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
