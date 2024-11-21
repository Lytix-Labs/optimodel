import json
import os

import boto3
from optimodel_server.OptimodelError import OptimodelError
from optimodel_server.Providers.CommonUtils import containsImageInMessages

from optimodel_types import AWSBedrockCredentials, ModelTypes
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

        # if jsonMode is True:
        #     logger.warn("JSON mode not supported for Bedrock")

        """
        @NOTE Bedrock does not currently support image types
        """
        if (
            model != ModelTypes.claude_3_haiku.name
            and model != ModelTypes.claude_3_5_sonnet.name
            and containsImageInMessages(messages)
        ):
            raise OptimodelError(
                "Bedrock with this model does not currently support image types"
            )

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
            case ModelTypes.llama_3_1_8b_instruct.name:
                modelId = "meta.llama3-1-8b-instruct-v1:0"
            case ModelTypes.llama_3_1_70b_instruct.name:
                modelId = "meta.llama3-1-70b-instruct-v1:0"
            case ModelTypes.llama_3_1_405b_instruct.name:
                modelId = "meta.llama3-1-405b-instruct-v1:0"
            case ModelTypes.claude_3_5_sonnet_20240620.name:
                modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            case ModelTypes.claude_3_5_sonnet_20241022.name:
                modelId = "anthropic.claude-3-5-sonnet-20241022-v2:0"
            case ModelTypes.claude_3_5_sonnet.name:
                modelId = "anthropic.claude-3-5-sonnet-20241022-v2:0"
            case ModelTypes.claude_3_sonnet.name:
                modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
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
                | ModelTypes.llama_3_1_8b_instruct.name
                | ModelTypes.llama_3_1_70b_instruct.name
                | ModelTypes.llama_3_1_405b_instruct.name
            ):
                finalPrompt = "<|begin_of_text|>"
                for index, message in enumerate(messages):
                    newlineChar = ""
                    if index != len(messages) - 1:
                        newlineChar = "\n"

                    if isinstance(message.content, str):
                        finalPrompt += f"<|start_header_id|>{message.role}<|end_header_id|>{newlineChar}{message.content}"
                    else:
                        # Find the text content of the message
                        textContent = next(
                            (x.text for x in message.content if x.type == "text"), None
                        )
                        finalPrompt += f"<|start_header_id|>{message.role}<|end_header_id|>{newlineChar}{textContent}"
                finalPrompt += (
                    "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
                )
                native_request = {
                    "prompt": finalPrompt,
                }
                if temperature is not None:
                    native_request["temperature"] = temperature
                if maxGenLen is not None:
                    native_request["max_gen_len"] = maxGenLen
            case (
                ModelTypes.claude_3_5_sonnet.name
                | ModelTypes.claude_3_haiku.name
                | ModelTypes.claude_3_5_sonnet_20240620.name
                | ModelTypes.claude_3_sonnet.name
            ):
                # Filter our system role from messages
                messagesNoSystem = []
                messageSystem = None
                for message in messages:
                    if message.role != "system":
                        if isinstance(message.content, str):
                            messagesNoSystem.append(
                                {
                                    "role": message.role,
                                    "content": [
                                        {"type": "text", "text": message.content}
                                    ],
                                }
                            )
                        else:
                            # We can pass images to claude via the same syntax
                            contentToUse = []
                            role = message.role
                            for content in message.content:
                                if content.type == "text":
                                    contentToUse.append(
                                        {"type": "text", "text": content.text}
                                    )
                                else:
                                    contentToUse.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": content.source.type,
                                                "media_type": content.source.mediaType,
                                                "data": content.source.data,
                                            },
                                        }
                                    )
                            messagesNoSystem.append(
                                {"role": role, "content": contentToUse}
                            )
                    else:
                        if isinstance(message.content, str):
                            messageSystem = message
                        else:
                            # We can pass images to claude via the same syntax
                            for content in message.content:
                                if content.type == "text":
                                    messageSystem = content.text
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": messagesNoSystem,
                }
                if temperature is not None:
                    native_request["temperature"] = temperature

                if maxGenLen is not None:
                    native_request["max_tokens"] = maxGenLen
                else:
                    """
                    Claud via bedrock requires this to be set
                    @see https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
                    """
                    native_request["max_tokens"] = 200000
                if messageSystem is not None:
                    native_request["system"] = messageSystem.content

            case (
                ModelTypes.mistral_7b_instruct.name
                | ModelTypes.mixtral_8x7b_instruct.name
            ):
                formattedMessage = f"<s> [INST] "
                for message in messages:
                    if isinstance(message.content, str):
                        formattedMessage += f"{message.content}"
                    else:
                        # Find the text content of the message
                        textContent = next(
                            (x for x in message.text if x.type == "text"), None
                        )
                        formattedMessage += f"{textContent.content}"
                formattedMessage += " [/INST]"
                native_request = {
                    "prompt": formattedMessage,
                    "temperature": temperature,
                }
                if maxGenLen is not None:
                    native_request["max_tokens"] = maxGenLen

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
                | ModelTypes.llama_3_1_8b_instruct.name
                | ModelTypes.llama_3_1_70b_instruct.name
                | ModelTypes.llama_3_1_405b_instruct.name
            ):
                response_text: str = model_response["generation"].strip().lstrip("\n")
                promptTokenCount = model_response["prompt_token_count"]
                generationTokenCount = model_response["generation_token_count"]

                # If response_text starts with  `<|end_header_id|>`, remove it
                if response_text.startswith("<|end_header_id|>"):
                    response_text = response_text[13:].lstrip("\n").strip()

                return QueryResponse(
                    modelOutput=response_text,
                    promptTokens=promptTokenCount,
                    generationTokens=generationTokenCount,
                )
            case (
                ModelTypes.claude_3_5_sonnet.name
                | ModelTypes.claude_3_haiku.name
                | ModelTypes.claude_3_5_sonnet_20240620.name
                | ModelTypes.claude_3_sonnet.name
            ):
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
