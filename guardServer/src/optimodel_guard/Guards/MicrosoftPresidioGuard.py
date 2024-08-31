import logging
from typing import List, Literal

from optimodel_guard.Guards.GuardBaseClass import GuardBaseClass, GuardEvalResponse
from optimodel_server_types import MicrosoftPresidioConfig, ModelMessage
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine


logger = logging.getLogger(__name__)


class MicrosoftPresidioGuard(GuardBaseClass):
    # Define which transformers model to use
    model_config = [
        {
            "lang_code": "en",
            "model_name": {
                "spacy": "en_core_web_lg",  # use a small spaCy model for lemmas, tokens etc.
                "transformers": "dslim/bert-base-NER",
            },
        }
    ]
    presidioNLPEngine = TransformersNlpEngine(models=model_config)
    presidioAnalyzer = AnalyzerEngine(nlp_engine=presidioNLPEngine)
    presidioAnonymizer = AnonymizerEngine()

    def handlePreQuery(
        self, messages: List[ModelMessage], config: MicrosoftPresidioConfig
    ) -> bool:
        logger.info(f"META_LLAMA_PROMPT_GUARD_86M is checking pre-query....")
        return self._evaluateUserMessages(messages, config, "user")

    def handlePostQuery(
        self, messages: List[ModelMessage], config: MicrosoftPresidioConfig
    ) -> bool:
        logger.info(f"META_LLAMA_PROMPT_GUARD_86M is checking post-query....")
        return self._evaluateUserMessages(messages, config, "assistant")

    def _evaluateUserMessages(
        self,
        messages: List[ModelMessage],
        config: MicrosoftPresidioConfig,
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

        # Now analyze with presidio
        results = self.presidioAnalyzer.analyze(
            text=messages, language="en", entities=config.entitiesToCheck
        )

        if len(results) > 0:
            logger.info(f"Found entities in microsoft presidio: {results}")
            return GuardEvalResponse(
                failure=True,
                metadata={"entitiesFound": [x.entity_type for x in results]},
            )

        return GuardEvalResponse(failure=False, metadata={})
