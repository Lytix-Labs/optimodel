from optimodel_server.Config import config
from optimodel_server_types import (
    QueryBody,
    SpeedPriority,
    TogetherAICredentials,
)
from optimodel_server.Config.types import SAAS_MODE


def getAllAvailableProviders(body: QueryBody):
    """
    First extract the model, and get all configs for it
    """
    allAvailableProviders = config.modelToProvider.get(body.modelToUse, [])

    """
    Filter out any that dont meet our maxGenLen passed
    """
    allAvailableProviders = [
        provider
        for provider in allAvailableProviders
        if provider["maxGenLen"] >= body.maxGenLen
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
        allAvailableProviders = filteredProviders

    # Bad luck, no providers for the model passed
    if len(allAvailableProviders) == 0:
        raise ValueError(
            f"Model {body.modelToUse} not found or nothing matches criteria (e.g. maxGenLen)"
        )

    return allAvailableProviders


def orderProviders(allAvailableProviders: list, body: QueryBody) -> list:
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
        Find the cheapest model
        """
        allAvailableProviders.sort(key=lambda x: x["pricePer1M"])
        return allAvailableProviders
