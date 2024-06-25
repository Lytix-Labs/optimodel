import enum
import json
from pydantic import BaseModel


class SpeedPriority(enum.Enum):
    low = "low"
    high = "high"


class ModelMessage(BaseModel):
    role: str
    content: str


class TogetherAICredentials(BaseModel):
    togetherApiKey: str


class QueryBody(BaseModel):
    messages: list[ModelMessage]
    modelToUse: str
    speedPriority: SpeedPriority | None = None
    temperature: float = 0.2
    maxGenLen: int = 1024

    """
    If we are running in SAAS mode, we'll expect each request to bring their
    own credentials
    """
    credentials: list[TogetherAICredentials] | None = None
