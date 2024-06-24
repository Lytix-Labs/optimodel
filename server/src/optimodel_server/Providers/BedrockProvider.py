import json
import os

import boto3

from optimodel_server.RequestTypes import ModelMessage
from optimodel_server.Config.types import ModelTypes
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
)


class BedrockProvider(BaseProviderClass):
    def __init__(self):
        self.stsClient = boto3.client("sts")
        self.bedrockClient = boto3.client(
            "bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        pass

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
        messages: list[ModelMessage],
        model: ModelTypes,
        temperature: int = 0.2,
        maxGenLen: int = 1024,
    ):
        match model:
            case ModelTypes.llama3_8b_instruct.name:
                modelId = "meta.llama3-8b-instruct-v1:0"
            case ModelTypes.llama3_70b_instruct.name:
                modelId = "meta.llama3-70b-instruct-v1:0"
            case _:
                raise Exception(f"Model {model} not supported")

        finalPrompt = "<|begin_of_text|>"
        for message in messages:
            finalPrompt += f"<|start_header_id|>{message.role}<|end_header_id|>{message.content}<|eot_id|>"
        finalPrompt += "<|start_header_id|>assistant<|end_header_id|>"

        native_request = {
            "prompt": finalPrompt,
            "max_gen_len": maxGenLen,
            "temperature": temperature,
        }
        # Convert the native request to JSON.
        request = json.dumps(native_request)

        response = self.bedrockClient.invoke_model(modelId=modelId, body=request)

        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        response_text = model_response["generation"]
        promptTokenCount = model_response["prompt_token_count"]
        generationTokenCount = model_response["generation_token_count"]

        return QueryResponse(
            modelOutput=response_text,
            promptTokens=promptTokenCount,
            generationTokens=generationTokenCount,
        )
