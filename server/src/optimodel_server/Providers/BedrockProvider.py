import json
import os

import boto3
from optimodel_server.OptimodelError import OptimodelError
from optimodel_server.Providers.CommonUtils import containsImageInMessages

from optimodel_server_types import AWSBedrockCredentials, ModelTypes
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
    QueryParams,
)


class BedrockProvider(BaseProviderClass):
    supportSAASMode = True
    supportJSONMode = False

    def __init__(self):
        if SAAS_MODE is None:
            self.stsClient = boto3.client("sts")
            self.bedrockClient = boto3.client(
                "bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")
            )

    def validateProvider(self):
        """
        Validate the account has access to AWS
        TODO: Make this check more robust by checking bedrock permissions directly
        """
        response = self.stsClient.get_caller_identity()
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            """
            Couldn't get caller identity from AWS
            """
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
            raise OptimodelError("JSON mode not supported for Bedrock")
        """
        @NOTE Bedrock does not currently support image types
        """
        if containsImageInMessages(messages):
            raise OptimodelError("Bedrock does not currently support image types")

        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise OptimodelError(
                    "Bedrock credentials not provided", provider="bedrock"
                )

            # Try to find the together credentials
            bedrockCreds = next(
                (x for x in credentials if type(x) == AWSBedrockCredentials), None
            )
            if bedrockCreds is None:
                # This should have been filtered out in the planner
                raise OptimodelError(
                    "Bedrock credentials not found", provider="bedrock"
                )

            session = boto3.Session(
                aws_access_key_id=bedrockCreds.awsAccessKeyId,
                aws_secret_access_key=bedrockCreds.awsSecretKey,
                region_name=bedrockCreds.awsRegion,
            )
            client = session.client(
                "bedrock-runtime", region_name=bedrockCreds.awsRegion
            )
        else:
            if self.bedrockClient is None:
                raise OptimodelError(
                    "Bedrock client not initialized", provider="bedrock"
                )
            client = self.bedrockClient

        match model:
            case ModelTypes.llama_3_8b_instruct.name:
                modelId = "meta.llama3-8b-instruct-v1:0"
            case ModelTypes.llama_3_70b_instruct.name:
                modelId = "meta.llama3-70b-instruct-v1:0"
            case ModelTypes.claude_3_5_sonnet.name:
                modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            case ModelTypes.claude_3_haiku.name:
                modelId = "anthropic.claude-3-haiku-20240307-v1:0"
            case ModelTypes.mistral_7b_instruct.name:
                modelId = "mistral.mistral-7b-instruct-v0:2"
            case ModelTypes.mixtral_8x7b_instruct.name:
                modelId = "mistral.mixtral-8x7b-instruct-v0:1"
            case _:
                raise OptimodelError(f"Model {model} not supported", provider="bedrock")

        match model:
            case (
                ModelTypes.llama_3_8b_instruct.name
                | ModelTypes.llama_3_70b_instruct.name
            ):
                finalPrompt = "<|begin_of_text|>"
                for message in messages:
                    finalPrompt += f"<|start_header_id|>{message.role}<|end_header_id|>{message.content}<|eot_id|>"
                finalPrompt += "<|start_header_id|>assistant<|end_header_id|>"
                native_request = {
                    "prompt": finalPrompt,
                }
                if temperature is not None:
                    native_request["temperature"] = temperature
                if maxGenLen is not None:
                    native_request["max_gen_len"] = maxGenLen
            case ModelTypes.claude_3_5_sonnet.name | ModelTypes.claude_3_haiku.name:
                # Check if we have a system prompt in messages
                systemPrompt = next((x for x in messages if x.role == "system"), None)

                # Filter our system role from messages
                messagesNoSystem = [
                    {
                        "role": x.role,
                        "content": [{"text": x.content, "type": "text"}],
                    }
                    for x in messages
                    if x.role != "system"
                ]
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": messagesNoSystem,
                }
                if temperature is not None:
                    native_request["temperature"] = temperature
                if maxGenLen is not None:
                    native_request["max_tokens"] = maxGenLen
                if systemPrompt is not None:
                    native_request["system"] = systemPrompt.content
            case (
                ModelTypes.mistral_7b_instruct.name
                | ModelTypes.mixtral_8x7b_instruct.name
            ):
                formattedMessage = f"<s> [INST] "
                for message in messages:
                    formattedMessage += f"{message.content}"
                formattedMessage += " [/INST]"
                native_request = {
                    "prompt": formattedMessage,
                    "max_tokens": maxGenLen,
                    "temperature": temperature,
                }

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        response = client.invoke_model(modelId=modelId, body=request)

        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        match model:
            case (
                ModelTypes.llama_3_8b_instruct.name
                | ModelTypes.llama_3_70b_instruct.name
            ):
                response_text = model_response["generation"].strip()
                promptTokenCount = model_response["prompt_token_count"]
                generationTokenCount = model_response["generation_token_count"]

                return QueryResponse(
                    modelOutput=response_text,
                    promptTokens=promptTokenCount,
                    generationTokens=generationTokenCount,
                )
            case ModelTypes.claude_3_5_sonnet.name | ModelTypes.claude_3_haiku.name:
                response_text = model_response["content"][0]["text"].strip()
                promptTokenCount = model_response["usage"]["input_tokens"]
                generationTokenCount = model_response["usage"]["output_tokens"]
                return QueryResponse(
                    modelOutput=response_text,
                    promptTokens=promptTokenCount,
                    generationTokens=generationTokenCount,
                )
            case (
                ModelTypes.mistral_7b_instruct.name
                | ModelTypes.mixtral_8x7b_instruct.name
            ):
                response_text = model_response["outputs"][0]["text"].strip()
                promptTokenCount = int(
                    response["ResponseMetadata"]["HTTPHeaders"][
                        "x-amzn-bedrock-input-token-count"
                    ]
                )
                generationTokenCount = int(
                    response["ResponseMetadata"]["HTTPHeaders"][
                        "x-amzn-bedrock-output-token-count"
                    ]
                )
                return QueryResponse(
                    modelOutput=response_text,
                    promptTokens=promptTokenCount,
                    generationTokens=generationTokenCount,
                )
