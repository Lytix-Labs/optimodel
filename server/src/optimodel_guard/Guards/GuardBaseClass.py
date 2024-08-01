from optimodel_server_types import GuardType
from optimodel_server_types.providerTypes import QueryParams, QueryResponse


class GuardBaseClass:
    """
    Common interface for all guards to implement
    """

    """
    Guards can intercept queries prior to making a query to the model, or evaluate
    after the query has been made to the model and before the response is returned
    """
    guardType: GuardType

    def handlePreQuery(self, query: QueryParams) -> bool:
        """
        Handle a pre-query event.

        @return bool: True if the query should be made, False if it should fail

        @NOTE This is only called for preQuery guards
        """
        pass

    def handlePostQuery(self, query: QueryParams, response: QueryResponse) -> bool:
        """
        Handle a post-query event.

        @return bool: True if the response should be returned to the user, False if it should fail

        @NOTE This is only called for postQuery guards
        """
        pass
