import enum
import json
from pydantic import BaseModel


class SpeedPriority(enum.Enum):
    low = "low"
    high = "high"


class ModelMessage(BaseModel):
    role: str
    content: str


class QueryBody(BaseModel):
    messages: list[ModelMessage]
    modelToUse: str
    speedPriority: SpeedPriority | None = None
