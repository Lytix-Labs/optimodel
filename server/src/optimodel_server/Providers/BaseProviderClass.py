from typing import List, TypedDict, Optional
from optimodel_server_types import (
    Credentials,
    ModelMessage,
    ModelTypes,
)


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


class QueryParams(TypedDict, total=False):
    messages: List[ModelMessage]
    model: ModelTypes
    temperature: float
    maxGenLen: Optional[int]
    credentials: Optional[List[Credentials]]
    jsonMode: Optional[bool]


class BaseProviderClass:
    """
    Base class for all providers. This serves as an interface for all providers to implement
    """

    """
    If the provider supports SAAS mode, set this to True
    """

    supportSAASMode: bool = False

    """
    If the provider supports JSON mode, set this to True
    """
    supportJSONMode: bool = False

    def validateProvider(self) -> bool:
        """
        Do we have the auth needed to validate this provider
        """
        pass

    def makeQuery(
        self,
        queryParams: QueryParams,
    ) -> QueryResponse:
        """
        Make a query to the provider given a model
        """
        pass
