import enum
import json
from pydantic import BaseModel


class ModelTypes(enum.Enum):
    # Meta
    llama_3_8b_instruct = "llama_3_8b_instruct"
    llama_3_70b_instruct = "llama_3_70b_instruct"

    # Anthropic
    claude_3_5_sonnet = "claude_3_5_sonnet"
    claude_3_haiku = "claude_3_haiku"

    # Mistral
    mistral_7b_instruct = "mistral_7b_instruct"
    mistral_8x7b_instruct = "mistral_8x7b_instruct"

    # OpenAI Related
    gpt_4 = "gpt_4"
    gpt_3_5_turbo = "gpt_3_5_turbo"
    gpt_4o = "gpt_4o"
    gpt_4_turbo = "gpt_4_turbo"
    gpt_3_5_turbo_0125 = "gpt_3_5_turbo_0125"


class SpeedPriority(enum.Enum):
    low = "low"
    high = "high"


class ModelMessage(BaseModel):
    role: str
    content: str


class TogetherAICredentials(BaseModel):
    togetherApiKey: str


class OpenAICredentials(BaseModel):
    openAiKey: str


class AWSBedrockCredentials(BaseModel):
    awsAccessKeyId: str
    awsSecretKey: str
    awsRegion: str


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
    credentials: (
        list[TogetherAICredentials | OpenAICredentials | AWSBedrockCredentials] | None
    ) = None
