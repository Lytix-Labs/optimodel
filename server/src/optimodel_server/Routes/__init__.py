from optimodel_types.providerTypes import GuardError
from pydantic import BaseModel
from typing import Dict, Optional, List
from optimodel_types import ModelMessage


class LytixProxyResponse(BaseModel):
    messages: Optional[List[Dict[str, str]]] = None
    messagesV2: Optional[List[ModelMessage]] = None
    inputTokens: Optional[int] = None
    outputTokens: Optional[int] = None
    cost: Optional[float] = None
    provider: Optional[str] = None
    guardErrors: Optional[List[GuardError]] = None
