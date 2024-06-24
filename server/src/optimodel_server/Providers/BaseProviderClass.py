from optimodel_server.Config.types import ModelTypes
from optimodel_server.RequestTypes import ModelMessage


class QueryResponse:
    def __init__(self, modelOutput: str, promptTokens: int, generationTokens: int):
        self.modelOutput = modelOutput
        self.promptTokens = promptTokens
        self.generationTokens = generationTokens

    modelOutput: str
    promptTokens: int
    generationTokens: int


class BaseProviderClass:
    def validateProvider(self) -> bool:
        """
        Do we have the auth needed to validate this provider
        """
        pass

    def makeQuery(
        self,
        messages: list[ModelMessage],
        model: ModelTypes,
        temperature: int = 0.2,
        maxGenLen: int = 1024,
    ) -> QueryResponse:
        """
        Make a query to the provider given a model
        """
        pass
