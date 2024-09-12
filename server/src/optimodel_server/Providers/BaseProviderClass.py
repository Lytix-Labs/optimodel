from optimodel_types.providerTypes import QueryParams, QueryResponse


class BaseProviderClass:
    """
    Base class for all providers. This serves as an interface for all providers to implement
    """

    """
    If the provider supports SAAS mode, set this to True
    """

    supportSAASMode: bool = False

    """
    If the provider supports JSON mode, set this to True
    """
    supportJSONMode: bool = False

    def validateProvider(self) -> bool:
        """
        Do we have the auth needed to validate this provider
        """
        pass

    def makeQuery(
        self,
        queryParams: QueryParams,
    ) -> QueryResponse:
        """
        Make a query to the provider given a model
        """
        pass
