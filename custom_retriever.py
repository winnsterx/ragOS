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
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from typing import Any, List
import logging
import re
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomRetriever(BaseRetriever):
    def __init__(self, retriever, llm, enable_reranker) -> None:
        self.retriever = retriever
        self.llm = llm
        self.enable_reranker = enable_reranker

    def parse_choice_select_answer_fn(self, answer, num_choices, raise_error=True):
        print("answer:", answer)
        answer_lines = answer.split("\n")
        answer_nums = []
        answer_relevances = []
        pattern = r"Document\s+(\d+).*?Relevance\s+score:\s+(\d+)"
        for answer_line in answer_lines:
            print("answer line", answer_line)
            match = re.search(pattern, answer_line)
            if match:
                answer_num = match.group(1)
                answer_relevance = match.group(2)
                answer_nums.append(answer_num)
                answer_relevances.append(float(answer_relevance))
                print("match: ", answer_num, answer_relevance)
            else:
                print("No answer found")

        return answer_nums, answer_relevances

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Try retrieving with RouterRetriever first,
        when router/selector throws error (ex: when it cant find a relevant index)
        catch error and return one empty node with empty score
        TODO: submit PR to selector to handle no indexs were found by LLM selector error
        """
        try:
            nodes = self.retriever._retrieve(query_bundle)
            for n in nodes:
                logger.info("node_str len: %s; node score: %s",
                            len(n.node.text), n.score)
            if self.enable_reranker:
                reranker = FlagEmbeddingReranker(
                    top_n=3,
                    model="BAAI/bge-reranker-large",
                    use_fp16=False,
                )
                nodes = reranker.postprocess_nodes(nodes, query_bundle)
                for n in nodes:
                    logger.info("POST_RERANKER: node_str len: %s; node score: %s",
                                len(n.node.text), n.score)
            return nodes
        except ValueError:
            logger.error(
                "Selector cannot find relevant index for question: %s", query_bundle)
            nodes = [NodeWithScore(node=TextNode(
                id_="no_index_selected", text="There is no nodes with relevant context"))]

            return nodes
