import json
import logging
import os

from optimodel_server.Providers.BaseProviderClass import BaseProviderClass
from optimodel_server.Providers import (
    BedrockProvider,
    GroqProvider,
    TogetherProvider,
    OpenAIProvider,
    AnthropicProvider,
)
from optimodel_server.Config.types import SAAS_MODE
from optimodel_server_types import ModelTypes


logger = logging.getLogger(__name__)


class Config:
    modelToProvider: dict[str, any]
    providerInstances: dict[str, BaseProviderClass]

    """
    Path to the config file. Try to use a custom one if present via 
    env var OPTIMODEL_CONFIG_PATH
    """
    configPath: str = os.environ.get(
        "OPTIMODEL_CONFIG_PATH",
        os.path.dirname(__file__) + "/../optimodel-server-config.json",
    )

    """
    URL to our guard server
    """
    guardServerURL: str = os.environ.get(
        "OPTIMODEL_GUARD_SERVER_URL", "http://localhost:8001"
    )

    def __init__(self):
        self.providerInstances = {}
        with open(self.configPath, "r") as config:
            """
            First lets validate the config
            """
            validatedConfig = self._validateConfig(json.load(config))

            self.config = validatedConfig

            """
            Set a mapping to model <-> provider for easier lookups in the future
            """
            self.modelToProvider = {}
            for provider, models in self.config["availableModels"].items():
                for model in models:
                    # Save the provider with this metadata
                    model["provider"] = provider

                    # Also save the price per token, so we dont need to ask the provider for it
                    model["costPerTokenInput"] = model["pricePer1MInput"] / 1_000_000
                    model["costPerTokenOutput"] = model["pricePer1MOutput"] / 1_000_000
                    if model["name"] in self.modelToProvider:
                        self.modelToProvider[model["name"]].append(model)
                    else:
                        self.modelToProvider[model["name"]] = [model]

    def _validateConfig(self, config: dict):
        """
        Validate the config
        """
        validatedConfig = {"availableModels": {}}
        for provider, models in config["availableModels"].items():
            """
            Make sure the model is valid
            """
            for model in models:
                if model["name"] not in ModelTypes:
                    print(model["name"], ModelTypes)
                    raise ValueError(f"Invalid model name: {model['name']}")

            """
            Make sure we've configured everything needed for this provider
            """
            providerClient: BaseProviderClass = None
            match provider:
                case "bedrock":
                    providerClient = BedrockProvider()
                case "together":
                    providerClient = TogetherProvider()
                case "openai":
                    providerClient = OpenAIProvider()
                case "groq":
                    providerClient = GroqProvider()
                case "anthropic":
                    providerClient = AnthropicProvider()
                case _:
                    logger.warn(f"Provider {provider} is not supported")
                    continue

            """
            In SAAS mode, each request will bring its own credentials
            No need to check it here
            """
            if SAAS_MODE is None:
                validProvider = providerClient.validateProvider()
            else:
                validProvider = providerClient.supportSAASMode

            if not validProvider:
                logger.warn(
                    f"Provider {provider} is not valid (e.g. credentials not configured etc.)"
                )
                continue

            """
            g2g, add to validated config and our provider instances
            """
            validatedConfig["availableModels"][provider] = models
            self.providerInstances[provider] = providerClient

        return validatedConfig
