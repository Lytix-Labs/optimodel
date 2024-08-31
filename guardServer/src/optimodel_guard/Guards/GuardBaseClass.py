from typing import Dict, Any
from optimodel_server_types.providerTypes import QueryParams, QueryResponse
from pydantic import BaseModel


class GuardEvalResponse(BaseModel):
    failure: bool
    metadata: Dict[str, Any] | None = {}


class GuardBaseClass:
    """
    Common interface for all guards to implement
    """

    def handlePreQuery(self, query: QueryParams) -> GuardEvalResponse:
        """
        Handle a pre-query event.

        @return bool: True if the query failed the check, False if it passed the check

        @NOTE This is only called for preQuery guards
        """
        pass

    def handlePostQuery(
        self, query: QueryParams, response: QueryResponse
    ) -> GuardEvalResponse:
        """
        Handle a post-query event.

        @return bool: True if the response failed the check, False if it passed the check

        @NOTE This is only called for postQuery guards
        """
        pass
