import enum
from typing import Literal
from pydantic import BaseModel


class ModelTypes(enum.Enum):
    # Meta
    llama_3_8b_instruct = "llama_3_8b_instruct"
    llama_3_70b_instruct = "llama_3_70b_instruct"
    llama_3_1_405b = "llama_3_1_405b"
    llama_3_1_70b = "llama_3_1_70b"
    llama_3_1_8b = "llama_3_1_8b"

    # Anthropic
    claude_3_5_sonnet = "claude_3_5_sonnet"
    claude_3_haiku = "claude_3_haiku"

    # Mistral
    mistral_7b_instruct = "mistral_7b_instruct"
    mixtral_8x7b_instruct = "mixtral_8x7b_instruct"

    # OpenAI Related
    gpt_4 = "gpt_4"
    gpt_3_5_turbo = "gpt_3_5_turbo"
    gpt_4o = "gpt_4o"
    gpt_4_turbo = "gpt_4_turbo"
    gpt_3_5_turbo_0125 = "gpt_3_5_turbo_0125"
    gpt_4o_mini = "gpt_4o_mini"


class Providers(enum.Enum):
    openai = "openai"
    togetherai = "togetherai"
    groq = "groq"
    anthropic = "anthropic"
    bedrock = "bedrock"


class SpeedPriority(enum.Enum):
    low = "low"
    high = "high"


class ModelImageMessageSource(BaseModel):
    type: str
    mediaType: str
    data: str


class ModelMessageContentEntry(BaseModel):
    type: str
    text: str | None = None
    source: ModelImageMessageSource | None = None


class ModelMessage(BaseModel):
    role: str
    content: str | list[ModelMessageContentEntry]


class TogetherAICredentials(BaseModel):
    togetherApiKey: str


class AnthropicCredentials(BaseModel):
    anthropicApiKey: str


class GroqCredentials(BaseModel):
    groqApiKey: str


class OpenAICredentials(BaseModel):
    openAiKey: str


class AWSBedrockCredentials(BaseModel):
    awsAccessKeyId: str
    awsSecretKey: str
    awsRegion: str


Credentials = (
    TogetherAICredentials
    | OpenAICredentials
    | AWSBedrockCredentials
    | GroqCredentials
    | AnthropicCredentials
)

GuardType = Literal["preQuery", "postQuery"]


class GuardQueryBase(BaseModel):
    """
    Base class for a guard query
    """

    guardType: GuardType


class LLamaPromptGuardConfig(GuardQueryBase):
    guardName: Literal["LLamaPromptGuard"]
    jailbreakThreshold: float | None = None
    injectionThreshold: float | None = None


Guards = LLamaPromptGuardConfig


class QueryBody(BaseModel):
    messages: list[ModelMessage]
    modelToUse: str
    speedPriority: SpeedPriority | None = None
    temperature: float = 0.2
    maxGenLen: int = None
    jsonMode: bool | None = None

    """
    Optionally select a provider to use for this request
    """
    provider: Providers | None = None

    """
    Optionally set a list of guards to check 
    """
    guards: list[Guards] | None = None

    """
    These are lytix specific options
    @ref https://docs.lytix.co/OptiModel/getting-started
    """
    userId: str | None = None
    sessionId: str | None = None

    """
    If we are running in SAAS mode, we'll expect each request to bring their
    own credentials
    """
    credentials: list[Credentials] | None = None


class GuardBody(BaseModel):
    messages: list[ModelMessage]
    modelOutput: str | None = None
    guard: Guards


class GuardResponse(BaseModel):
    """
    Response from the guard endpoint
    """

    """
    Map of guard name to whether it passed or failed
    """
    failure: bool
