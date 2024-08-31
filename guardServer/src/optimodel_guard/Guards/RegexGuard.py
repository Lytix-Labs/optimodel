import logging
import re
from typing import List, Literal


from optimodel_guard.Guards.GuardBaseClass import GuardBaseClass, GuardEvalResponse
from optimodel_server_types import (
    LytixRegexConfig,
    ModelMessage,
)


logger = logging.getLogger(__name__)


class LytixRegexGuard(GuardBaseClass):
    def handlePreQuery(
        self, messages: List[ModelMessage], config: LytixRegexConfig
    ) -> bool:
        logger.info(f"LYTIX_REGEX_GUARD is checking pre-query....")
        return self._evaluateRegexMessage(messages, config, "user")

    def handlePostQuery(
        self, messages: List[ModelMessage], config: LytixRegexConfig
    ) -> bool:
        logger.info(f"LYTIX_REGEX_GUARD is checking post-query....")
        return self._evaluateRegexMessage(messages, config, "assistant")

    def _evaluateRegexMessage(
        self,
        messages: List[ModelMessage],
        config: LytixRegexConfig,
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

        try:
            pattern = re.compile(config.regex)
            match = pattern.search(messages)
        except re.error as e:
            logger.error(f"Error compiling regex pattern: {e}")
            return GuardEvalResponse(failure=False, metadata={})

        if match:
            logger.info(f"Regex pattern matched: {match.group()}")
            return GuardEvalResponse(failure=True, metadata={"matched": match.group()})
        else:
            logger.info("No match found for the regex pattern.")

        return GuardEvalResponse(failure=False, metadata={})
