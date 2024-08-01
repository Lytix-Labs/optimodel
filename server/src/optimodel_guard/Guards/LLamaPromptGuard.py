import logging
from typing import List, Literal

from optimodel_guard.Guards.GuardBaseClass import GuardBaseClass
from optimodel_server_types import LLamaPromptGuardConfig, ModelMessage
from optimodel_server_types.providerTypes import QueryParams, QueryResponse

from transformers import pipeline


logger = logging.getLogger(__name__)


class LLamaPromptGuard(GuardBaseClass):
    classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")

    def handlePreQuery(
        self, messages: List[ModelMessage], config: LLamaPromptGuardConfig
    ) -> bool:
        logger.info(f"LLamaPromptGuard is checking pre-query....")
        return self._evaluateUserMessages(messages, config, "user")

    def handlePostQuery(
        self, messages: List[ModelMessage], config: LLamaPromptGuardConfig
    ) -> bool:
        logger.info(f"LLamaPromptGuard is checking post-query....")
        return self._evaluateUserMessages(messages, config, "assistant")

    def _evaluateUserMessages(
        self,
        messages: List[ModelMessage],
        config: LLamaPromptGuardConfig,
        role: Literal["user", "assistant"],
    ) -> bool:
        """
        Extract any instructions from the query that the user has given.
        """
        messages = ",".join(
            [message.content for message in messages if message.role == role]
        )
        results = self.classifier(messages)
        logger.info(f"LLamaPromptGuard results: {results} with {messages}")

        # Pull out `INJECTION` and/or `JAILBREAK` from the response
        injectionList = [
            result["score"] for result in results if result["label"] == "INJECTION"
        ]
        jailbreakList = [
            result["score"] for result in results if result["label"] == "JAILBREAK"
        ]

        if (
            len(injectionList) > 0
            and config.injectionThreshold
            and config.injectionThreshold < max(injectionList)
        ):
            logger.info(
                f"LLamaPromptGuard found injection. Failed {role} check. Got {max(injectionList)}"
            )
            return True
        if (
            len(jailbreakList) > 0
            and config.jailbreakThreshold
            and config.jailbreakThreshold < max(jailbreakList)
        ):
            logger.info(
                f"LLamaPromptGuard found jailbreak. Failed {role} check. Got {max(jailbreakList)}"
            )
            return True

        return False
