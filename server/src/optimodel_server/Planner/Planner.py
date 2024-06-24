from optimodel_server.Config import config
from optimodel_server.RequestTypes import QueryBody, SpeedPriority


def getAllAvailableProviders(body: QueryBody):
    """
    First extract the model, and get all configs for it
    """
    allAvailableProviders = config.modelToProvider.get(body.modelToUse, [])

    # Bad luck, no providers for the model passed
    if len(allAvailableProviders) == 0:
        raise ValueError(f"Model {body.modelToUse} not found")

    return allAvailableProviders


def orderProviders(allAvailableProviders: list, body: QueryBody) -> list:
    """
    Now we have a list of providers, and we need to pick the best one by lining them up in order of priority
    Heres our gameplan
    1. Speed is most important to us, if the user says they want the fastest, always pick the fastest
    2. Otherwise default to cost, we'd like to save money as much as we can
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
