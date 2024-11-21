import json
import logging
import os

from optimodel_server.Providers.BaseProviderClass import BaseProviderClass
from optimodel_server.Providers import (
    BedrockProvider,
    GroqProvider,
    MistralCodestralProvider,
    TogetherProvider,
    OpenAIProvider,
    AnthropicProvider,
    MistralAIProvider,
    GeminiProvider,
)
from optimodel_server.Config.types import SAAS_MODE
from optimodel_types import ModelTypes
import requests


logger = logging.getLogger(__name__)

# URL of the JSON file
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"


class Config:
    modelToProvider: dict[str, any]
    providerInstances: dict[str, BaseProviderClass]
    liteLLMPricing: dict[str, any]
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
        Attempt to pull the liteLLM pricing json
        """
        response = requests.get(LITELLM_PRICING_URL)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON content
            data = response.json()
            self.liteLLMPricing = data
        else:
            logger.error(
                f"Failed to retrieve the liteLLM pricing json. Status code: {response.status_code}"
            )
            raise Exception(
                f"Failed to retrieve the liteLLM pricing json. Status code: {response.status_code}"
            )

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
                case "mistralai":
                    providerClient = MistralAIProvider()
                case "mistralcodestral":
                    providerClient = MistralCodestralProvider()
                case "gemini":
                    providerClient = GeminiProvider()
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
            Now lets see if we have a liteLLM index for this model
            """
            for model in models:
                if model.get("liteLLMIndex") is None:
                    if (
                        model.get("pricePer1MInput") is None
                        and model.get("pricePer1MOutput") is None
                    ):
                        logger.error(
                            f"LiteLLM index {model['name']} not found, and no pricePer1MInput or pricePer1MOutput provided"
                        )
                        continue
                else:
                    """
                    We need to pull the pricePer1MInput and pricePer1MOutput from the liteLLM pricing json
                    """
                    costPerTokenInput = self.liteLLMPricing[model["liteLLMIndex"]][
                        "input_cost_per_token"
                    ]
                    costPerTokenOutput = self.liteLLMPricing[model["liteLLMIndex"]][
                        "output_cost_per_token"
                    ]
                    model["pricePer1MInput"] = costPerTokenInput * 1_000_000
                    model["pricePer1MOutput"] = costPerTokenOutput * 1_000_000

                    if (
                        self.liteLLMPricing[model["liteLLMIndex"]].get(
                            "input_cost_per_token_above_128k_tokens"
                        )
                        is not None
                        and self.liteLLMPricing[model["liteLLMIndex"]].get(
                            "output_cost_per_token_above_128k_tokens"
                        )
                        is not None
                    ):
                        """
                        Also gemini is weird, save the pricePer1M above
                        """
                        costPerTokenAbove125KInput = self.liteLLMPricing[
                            model["liteLLMIndex"]
                        ]["input_cost_per_token_above_128k_tokens"]
                        costPerTokenAbove125KOutput = self.liteLLMPricing[
                            model["liteLLMIndex"]
                        ]["output_cost_per_token_above_128k_tokens"]
                        model["pricePer1MInputAbove125K"] = (
                            costPerTokenAbove125KInput * 1_000_000
                        )
                        model["pricePer1MOutputAbove125K"] = (
                            costPerTokenAbove125KOutput * 1_000_000
                        )
            """
            g2g, add to validated config and our provider instances
            """
            validatedConfig["availableModels"][provider] = models
            self.providerInstances[provider] = providerClient

        return validatedConfig
