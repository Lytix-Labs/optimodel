import logging

from optimodel_server.Config import config
from optimodel_server.OptimodelError import OptimodelError
from optimodel_types import (
    GeminiCredentials,
    GroqCredentials,
    MistralAICredentials,
    MistralCodeStralCredentials,
    QueryBody,
    SpeedPriority,
    TogetherAICredentials,
    AWSBedrockCredentials,
    OpenAICredentials,
    AnthropicCredentials,
    Providers,
)
from optimodel_server.Config.types import SAAS_MODE


logger = logging.getLogger(__name__)


def getAllAvailableProviders(body: QueryBody):
    """
    First extract the model, and get all configs for it
    """
    allAvailableProviders = config.modelToProvider.get(body.modelToUse, [])

    """
    If we've explicitly passed a provider, filter our list to only that provider
    """
    logger.info(f"allAvailableProviders: {allAvailableProviders}")

    if body.provider is not None:
        allAvailableProviders = [
            provider
            for provider in allAvailableProviders
            if provider["provider"] == body.provider.name
        ]

    """
    If we are running in SAAS mode, filter our any that we dont have creds for
    """

    if SAAS_MODE is not None:
        if body.credentials is None:
            raise ValueError("No credentials provided")
        filteredProviders = []
        for provider in allAvailableProviders:
            # Make sure we have the creds for this provider
            if provider["provider"] == "together":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == TogetherAICredentials),
                    None,
                )
                if credsForProvider is not None:
                    # Cool we have the creds for this provider, its valid 🙌
                    filteredProviders.append(provider)

            if provider["provider"] == "bedrock":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == AWSBedrockCredentials),
                    None,
                )
                if credsForProvider is not None:
                    # Cool we have the creds for this provider, its valid 🙌
                    filteredProviders.append(provider)

            if provider["provider"] == "openai":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == OpenAICredentials),
                    None,
                )
                if credsForProvider is not None:
                    filteredProviders.append(provider)

            if provider["provider"] == "groq":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == GroqCredentials),
                    None,
                )
                if credsForProvider is not None:
                    filteredProviders.append(provider)

            if provider["provider"] == "anthropic":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == AnthropicCredentials),
                    None,
                )
                if credsForProvider is not None:
                    filteredProviders.append(provider)

            if provider["provider"] == "mistralai":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == MistralAICredentials),
                    None,
                )
                if credsForProvider is not None:
                    filteredProviders.append(provider)

            if provider["provider"] == "mistralcodestral":
                credsForProvider = next(
                    (
                        x
                        for x in body.credentials
                        if type(x) == MistralCodeStralCredentials
                    ),
                    None,
                )
                if credsForProvider is not None:
                    filteredProviders.append(provider)

            if provider["provider"] == "gemini":
                credsForProvider = next(
                    (x for x in body.credentials if type(x) == GeminiCredentials),
                    None,
                )
                if credsForProvider is not None:
                    filteredProviders.append(provider)

        allAvailableProviders = filteredProviders

    # Bad luck, no providers for the model passed
    if len(allAvailableProviders) == 0:
        raise OptimodelError(
            f"Model {body.modelToUse} not found or nothing matches criteria (e.g. no valid credentials available)"
        )

    return allAvailableProviders


def orderProviders(allAvailableProviders: list[Providers], body: QueryBody) -> list:
    """
    Now we have a list of providers, and we need to pick the best one by lining them up in order of priority
    Heres our gameplan
    - Speed is most important to us, if the user says they want the fastest, always pick the fastest
    - Otherwise default to cost, we'd like to save money as much as we can
    """
    if body.speedPriority == SpeedPriority.high:
        """
        Find the fastest model
        """
        allAvailableProviders.sort(key=lambda x: x["speed"])
        return allAvailableProviders
    else:
        """
        Find the cheapest model by avg the input/output price
        """
        allAvailableProviders.sort(
            key=lambda x: (x["pricePer1MInput"] + x["pricePer1MOutput"]) / 2
        )
        return allAvailableProviders
