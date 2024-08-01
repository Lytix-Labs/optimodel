import logging

from optimodel_guard.Guards.GuardBaseClass import GuardBaseClass
from optimodel_server_types.providerTypes import QueryParams, QueryResponse


logger = logging.getLogger(__name__)


class ExamplePlugin(GuardBaseClass):
    pluginType = "preQuery"

    def handlePreQuery(self, query: QueryParams) -> bool:
        logger.info(f"ExamplePlugin is working....")
        return True

    def handlePostQuery(self, query: QueryParams, response: QueryResponse) -> bool:
        logger.info(f"ExamplePlugin is working....")
        return True
