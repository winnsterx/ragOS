"""
Problem: LLM selector in RouterRetriever is returning
zero retrievers (none of the retrievers fit).

NOT: a retriever retriving zero nodes.

Solution: build custom retriever that
1. uses RouterRetriever as default
2. catches the parse error
3. return custom node that says "no answer"
4. put retrieved nodes into LLM chat engine

"""
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever, RouterRetriever
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore
from typing import Any, List
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomRetriever(BaseRetriever):
    def __init__(self, router_retriever) -> None:
        self.router_retriever = router_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Try retrieving with RouterRetriever first,
        if throwing error, then return Node with score

        chat engine -> router retriever -> selector (finds nothing, returns [] instead of error)
        """
        try:
            nodes = self.router_retriever._retrieve(query_bundle)
            return nodes
        except ValueError:
            logger.error("Failed to retrieve bc of ValueError")
