import enum
import json
import logging
import os

from optimodel_server.Providers.BaseProviderClass import BaseProviderClass
from optimodel_server.Providers import BedrockProvider, TogetherProvider
from optimodel_server.Config.types import ModelTypes


logger = logging.getLogger(__name__)


class Config:
    modelToProvider: dict[str, any]
    providerInstances: dict[str, BaseProviderClass]

    def __init__(self):
        self.providerInstances = {}
        with open(
            (os.path.dirname(__file__) + "/../../optimodel-server-config.json"), "r"
        ) as config:
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
                    raise ValueError(f"Invalid model name: {model['name']}")

            """
            Make sure we've configured everything needed for this provider
            """
            providerClient: BaseProviderClass = None
            match provider:
                case "aws-bedrock":
                    providerClient = BedrockProvider()
                case "together":
                    providerClient = TogetherProvider()
                case _:
                    logger.warn(f"Provider {provider} is not supported")
                    continue

            validProvider = providerClient.validateProvider()
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
