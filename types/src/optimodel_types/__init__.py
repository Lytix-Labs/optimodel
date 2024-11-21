import enum
from typing import Literal, Dict, Any
from pydantic import BaseModel


class ModelTypes(enum.Enum):
    # Meta
    llama_3_8b_instruct = "llama_3_8b_instruct"
    llama_3_70b_instruct = "llama_3_70b_instruct"
    llama_3_1_405b = "llama_3_1_405b"
    llama_3_1_70b = "llama_3_1_70b"
    llama_3_1_8b = "llama_3_1_8b"
    llama_3_1_405b_instruct = "llama_3_1_405b_instruct"
    llama_3_1_70b_instruct = "llama_3_1_70b_instruct"
    llama_3_1_8b_instruct = "llama_3_1_8b_instruct"

    # Anthropic
    claude_3_5_sonnet_20240620 = "claude_3_5_sonnet_20240620"
    claude_3_5_sonnet_20241022 = "claude_3_5_sonnet_20241022"
    claude_3_haiku_20240307 = "claude_3_haiku_20240307"
    claude_3_5_sonnet = "claude_3_5_sonnet"
    claude_3_haiku = "claude_3_haiku"
    claude_3_sonnet = "claude_3_sonnet"

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
    gpt_4o_mini_2024_07_18 = "gpt_4o_mini_2024_07_18"
    gpt_4o_2024_08_06 = "gpt_4o_2024_08_06"
    gpt_4o_2024_05_13 = "gpt_4o_2024_05_13"
    o1_preview = "o1_preview"
    o1_preview_2024_09_12 = "o1_preview_2024_09_12"
    o1_mini = "o1_mini"
    o1_mini_2024_09_12 = "o1_mini_2024_09_12"

    # MistralAI
    codestral_latest = "codestral_latest"
    mistral_large_latest = "mistral_large_latest"
    open_mistral_nemo = "open_mistral_nemo"

    # Google
    gemini_1_5_pro = "gemini_1_5_pro"
    gemini_1_5_pro_latest = "gemini_1_5_pro_latest"
    gemini_1_5_pro_001 = "gemini_1_5_pro_001"
    gemini_1_5_pro_002 = "gemini_1_5_pro_002"
    gemini_1_5_pro_exp_0801 = "gemini_1_5_pro_exp_0801"
    gemini_1_5_pro_exp_0827 = "gemini_1_5_pro_exp_0827"
    gemini_1_5_flash = "gemini_1_5_flash"
    gemini_1_5_flash_latest = "gemini_1_5_flash_latest"
    gemini_1_5_flash_001 = "gemini_1_5_flash_001"
    gemini_1_5_flash_001_tuning = "gemini_1_5_flash_001_tuning"
    gemini_1_5_flash_exp_0827 = "gemini_1_5_flash_exp_0827"
    gemini_1_5_flash_8b_exp_0827 = "gemini_1_5_flash_8b_exp_0827"
    gemini_1_5_flash_8b = "gemini_1_5_flash_8b"


class Providers(enum.Enum):
    openai = "openai"
    togetherai = "togetherai"
    groq = "groq"
    anthropic = "anthropic"
    bedrock = "bedrock"
    gemini = "gemini"
    mistralai = "mistralai"
    mistralcodestral = "mistralcodestral"


class SpeedPriority(enum.Enum):
    low = "low"
    high = "high"


class ModelImageMessageSource(BaseModel):
    type: str
    mediaType: str
    data: str


class GeminiFileData(BaseModel):
    fileUri: str
    mimeType: str


class ModelMessageContentEntry(BaseModel):
    type: str
    text: str | None = None
    source: ModelImageMessageSource | None = None
    data: GeminiFileData | None = None


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


class MistralAICredentials(BaseModel):
    mistralApiKey: str


class MistralCodeStralCredentials(BaseModel):
    mistralCodeStralApiKey: str


class GeminiCredentials(BaseModel):
    geminiApiKey: str


Credentials = (
    TogetherAICredentials
    | OpenAICredentials
    | AWSBedrockCredentials
    | GroqCredentials
    | AnthropicCredentials
    | MistralAICredentials
    | MistralCodeStralCredentials
    | GeminiCredentials
)

GuardType = Literal["preQuery", "postQuery"]


class GuardQueryBase(BaseModel):
    """
    Base class for a guard query
    """

    guardType: GuardType
    blockRequest: bool = False
    blockRequestMessage: str | None = None


class LLamaPromptGuardConfig(GuardQueryBase):
    guardName: Literal["META_LLAMA_PROMPT_GUARD_86M"]
    jailbreakThreshold: float | None = None
    injectionThreshold: float | None = None


class LytixRegexConfig(GuardQueryBase):
    guardName: Literal["LYTIX_REGEX_GUARD"]
    regex: str


class MicrosoftPresidioConfig(GuardQueryBase):
    guardName: Literal["MICROSOFT_PRESIDIO_GUARD"]
    entitiesToCheck: list[str]


Guards = LLamaPromptGuardConfig | LytixRegexConfig | MicrosoftPresidioConfig


class QueryBody(BaseModel):
    messages: list[ModelMessage]
    modelToUse: str
    speedPriority: SpeedPriority | None = None
    temperature: float | None = None
    maxGenLen: int | None = None
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

    """
    Metadata associated with the failure
    """
    metadata: Dict[str, Any] | None = {}
