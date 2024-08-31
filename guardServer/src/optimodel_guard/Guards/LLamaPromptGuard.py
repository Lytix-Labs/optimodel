import logging
from typing import List, Literal

from optimodel_guard.Guards.GuardBaseClass import GuardBaseClass, GuardEvalResponse
from optimodel_server_types import LLamaPromptGuardConfig, ModelMessage
from optimodel_server_types.providerTypes import QueryParams, QueryResponse

from transformers import pipeline


logger = logging.getLogger(__name__)


class LLamaPromptGuard(GuardBaseClass):
    classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")

    def handlePreQuery(
        self, messages: List[ModelMessage], config: LLamaPromptGuardConfig
    ) -> bool:
        logger.info(f"META_LLAMA_PROMPT_GUARD_86M is checking pre-query....")
        return self._evaluateUserMessages(messages, config, "user")

    def handlePostQuery(
        self, messages: List[ModelMessage], config: LLamaPromptGuardConfig
    ) -> bool:
        logger.info(f"META_LLAMA_PROMPT_GUARD_86M is checking post-query....")
        return self._evaluateUserMessages(messages, config, "assistant")

    def _evaluateUserMessages(
        self,
        messages: List[ModelMessage],
        config: LLamaPromptGuardConfig,
        role: Literal["user", "assistant"],
    ) -> GuardEvalResponse:
        """
        Extract any instructions from the query that the user has given.
        """
        relatedMessages = [message for message in messages if message.role == role]
        messagesRaw = []

        for message in relatedMessages:
            if isinstance(message.content, str):
                messagesRaw.append(message.content)
            else:
                for entry in message.content:
                    if isinstance(entry, str):
                        messagesRaw.append(entry)
                    elif entry.type == "text":
                        messagesRaw.append(entry.text)

        messages = ",".join(messagesRaw)
        results = self.classifier(messages)

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
                f"META_LLAMA_PROMPT_GUARD_86M found injection. Failed {role} check. Got {max(injectionList)}"
            )
            return GuardEvalResponse(
                failure=True, metadata={"injectionScore": max(injectionList)}
            )
        if (
            len(jailbreakList) > 0
            and config.jailbreakThreshold
            and config.jailbreakThreshold < max(jailbreakList)
        ):
            logger.info(
                f"META_LLAMA_PROMPT_GUARD_86M found jailbreak. Failed {role} check. Got {max(jailbreakList)}"
            )
            return GuardEvalResponse(
                failure=True, metadata={"jailbreakScore": max(jailbreakList)}
            )

        return GuardEvalResponse(failure=False, metadata={})
