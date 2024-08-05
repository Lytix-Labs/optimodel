from .LLamaPromptGuard import LLamaPromptGuard
from .RegexGuard import LytixRegexGuard
from .MicrosoftPresidioGuard import MicrosoftPresidioGuard

GuardMapping = {
    "META_LLAMA_PROMPT_GUARD_86M": LLamaPromptGuard(),
    "LYTIX_REGEX_GUARD": LytixRegexGuard(),
    "MICROSOFT_PRESIDIO_GUARD": MicrosoftPresidioGuard(),
}
