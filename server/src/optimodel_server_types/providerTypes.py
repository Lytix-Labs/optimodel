from typing import List, Optional, TypedDict
from optimodel_server_types import Credentials, ModelMessage, ModelTypes, Providers


class QueryResponse:
    """
    Response from a query to the provider
    """

    def __init__(self, modelOutput: str, promptTokens: int, generationTokens: int):
        self.modelOutput = modelOutput
        self.promptTokens = promptTokens
        self.generationTokens = generationTokens

    modelOutput: str
    promptTokens: int
    generationTokens: int


class MakeQueryResponse(TypedDict):
    """
    Response from a query to the provider
    """

    modelResponse: str
    promptTokens: int
    generationTokens: int
    cost: float
    provider: Providers
    guardErrors: List[str]


class QueryParams(TypedDict, total=False):
    messages: List[ModelMessage]
    model: ModelTypes
    temperature: float
    maxGenLen: Optional[int]
    credentials: Optional[List[Credentials]]
    jsonMode: Optional[bool]
