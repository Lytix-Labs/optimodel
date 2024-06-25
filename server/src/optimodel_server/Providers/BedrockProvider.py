import json
import os

import boto3

from optimodel_server_types import ModelMessage, AWSBedrockCredentials, ModelTypes
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server.Providers.BaseProviderClass import (
    BaseProviderClass,
    QueryResponse,
)


class BedrockProvider(BaseProviderClass):
    supportSAASMode = False

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
        messages: list[ModelMessage],
        model: ModelTypes,
        temperature: int = 0.2,
        maxGenLen: int = 1024,
        credentials: AWSBedrockCredentials | None = None,
    ):
        """
        Currently bedrock does not support SAAS mode
        """
        if SAAS_MODE is not None:
            if credentials is None:
                # This should have been filtered out in the planner
                raise Exception("Together credentials not provided")

            # Try to find the together credentials
            bedrockCreds = next(
                (x for x in credentials if type(x) == AWSBedrockCredentials), None
            )
            if bedrockCreds is None:
                # This should have been filtered out in the planner
                raise Exception("Bedrock credentials not found")

            session = boto3.Session(
                aws_access_key_id=bedrockCreds.awsAccessKeyId,
                aws_secret_access_key=bedrockCreds.awsSecretKey,
            )
            client = session.client("bedrock-runtime")
        else:
            if self.bedrockClient is None:
                raise Exception("Bedrock client not initialized")
            client = self.bedrockClient

        if SAAS_MODE is not None:
            raise Exception("Bedrock does not support SAAS mode")

        match model:
            case ModelTypes.llama_3_8b_instruct.name:
                modelId = "meta.llama3-8b-instruct-v1:0"
            case ModelTypes.llama_3_70b_instruct.name:
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

        response = client.invoke_model(modelId=modelId, body=request)

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
