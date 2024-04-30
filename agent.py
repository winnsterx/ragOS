from custom_retriever import CustomRetriever
import chromadb
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, SummaryIndex, Document, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.tools import QueryEngineTool, RetrieverTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.retrievers import RouterRetriever, VectorIndexRetriever
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine, CondenseQuestionChatEngine
from llama_index.llms.ollama import Ollama
from llama_index.readers.notion import NotionPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from bs4 import BeautifulSoup
import time
import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


# Settings need to be
# - accessible by all methods inside this class, like a global config
# - different for each new instance of the class. agent B and agent A can have different Settings, changing one doesnt affect the other
class Agent:
    # Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    # Settings.llm = Ollama(model="llama3", request_timeout=100)

    def __init__(self, llm_model="llama3", embed_model="local:Salesforce/SFR-Embedding-Mistral", mode="context", system_prompt="You are a helpful, calm, succinct chatbot. Please always use the tools provided to answer a question. Do not rely on prior knowledge."):
        self.db = chromadb.PersistentClient(path="./chroma")
        self.llm = Ollama(model=llm_model, request_timeout=100)
        self.embed_model = embed_model
        self.mode = mode
        self.system_prompt = system_prompt
        self.update_router_with_collections()

    def update_router_with_collections(self):
        collections = self.db.list_collections()
        if self.mode == "condense_question":
            router_query_engine = self.build_router_query_engine(collections)
            self.chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=router_query_engine, llm=self.llm, verbose=True)
        elif self.mode == "context":
            router_retriever = self.build_router_retriever(collections)
            custom_retriever = CustomRetriever(router_retriever, self.llm)
            self.chat_engine = ContextChatEngine.from_defaults(
                retriever=custom_retriever, llm=self.llm, verbose=True, system_prompt=self.system_prompt)

    def scrape_web_pages(self, url, max_depth=2):
        start = time.process_time()
        loader = RecursiveUrlLoader(
            url=url, max_depth=max_depth, extractor=lambda x: BeautifulSoup(x, "html.parser").get_text())
        pages = loader.load()

        logger.info("Took %s seconds to load %s URLs",
                    time.process_time() - start, len(pages))
        documents = [Document(text=p.page_content, metadata=p.metadata)
                     for p in pages]
        return documents

    def create_index_from_notion_page(self, page_id, name, description):
        notion_integration_token = os.getenv("NOTION_INTEGRATIONS_TOKEN")
        documents = NotionPageReader(
            integration_token=notion_integration_token).load_data(page_ids=[page_id])
        print("Notion docs", documents)
        self.create_collection(
            name, documents, description)

    def load_collection(self, name):
        """
        Loads existing collection with name from chroma db and
        creates a vector store index from it
        """
        collection = self.db.get_or_create_collection(name=name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=self.embed_model)
        return index

    def create_collection(self, name, documents, description):
        """
        Creates a new collection in Chroma persistent DB,
        creates a vector index from new documents,
        stores the index in storage by changing the default to storage context
        """
        collection = self.db.get_or_create_collection(
            name=name, metadata={"description": description})
        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents, storage_context=storage_context, embed_model=self.embed_model)

        return index

    def build_router_query_engine(self, collections):
        tools = []
        for c in collections:
            index = self.load_collection(c.name)
            tool = QueryEngineTool.from_defaults(
                query_engine=index.as_query_engine(llm=self.llm, streaming=True), description=c.metadata["description"])
            logger.info("Tool: %s", tool.metadata)
            tools.append(tool)
        router_query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(llm=self.llm), query_engine_tools=tools, llm=self.llm, verbose=True)
        return router_query_engine

    def build_router_retriever(self, collections):
        tools = []
        for c in collections:
            index = self.load_collection(c.name)
            # retriever = VectorIndexRetriever(
            #     index=index, similarity_top_k=3, embed_model=self.embed_model, )
            tool = RetrieverTool.from_defaults(
                retriever=index.as_retriever(), description=c.metadata["description"])
            logger.info("Tool: %s", tool.metadata)
            tools.append(tool)
        router_retriever = RouterRetriever.from_defaults(
            retriever_tools=tools, llm=self.llm, selector=LLMMultiSelector.from_defaults(llm=self.llm))

        """
        need to build a custom retriever that 
        if res := router_retriever.retrieve() == []:
            res = "nothing relevant in the database" 
        """
        return router_retriever

    def create_index_from_url(self, url: str, max_depth: int, collection_name: str, collection_description: str) -> None:
        documents = self.scrape_web_pages(url, max_depth)
        self.create_collection(
            collection_name, documents, collection_description)

    def create_index_from_file(self, file_text: str, collection_name: str, collection_description: str) -> None:
        document = Document(text=file_text, metadata={
            "description": collection_description, "name": collection_name})
        self.create_collection(
            collection_name, [document], collection_description)
