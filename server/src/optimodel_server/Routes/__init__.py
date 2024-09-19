from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class LytixProxyResponse(BaseModel):
    messages: List[Dict[str, str]]
    inputTokens: Optional[int] = None
    outputTokens: Optional[int] = None
