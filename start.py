import chromadb
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
import time


def get_web_pages(url="https://paulgraham.com/"):
    start = time.process_time()
    loader = RecursiveUrlLoader(
        url=url, max_depth=3, extractor=lambda x: BeautifulSoup(x, "html.parser").get_text())
    pages = loader.load()
    print("Took", time.process_time() - start,
          "seconds to load", len(pages), "URLs")
    documents = [Document(text=p.page_content, metadata=p.metadata)
                 for p in pages]

    return documents


def create_and_store_index_in_nosql(documents):
    # VectorStoreIndex uses global embed_model and LLM specified above to create vectors and relationships of nodes
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="indexes/")
    return index


def load_index_from_nosql():
    # loads the VectorStoreIndex from local memory

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./indexes"),
        vector_store=SimpleVectorStore.from_persist_dir(
            persist_dir="./indexes"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./indexes"))
    index = load_index_from_storage(storage_context=storage_context)
    return index


def load_index_from_chroma():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("first_rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context)
    return index


def create_and_store_index_in_chroma(documents):
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("first_rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # this loads it and saves it to storage context
    index = VectorStoreIndex.from_documents(
        documents, storage_context)
    return index


def create_vector_index(fresh_scrape, documents=[]):
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    Settings.llm = Ollama(model="llama2", request_timeout=100)
    index = None

    if fresh_scrape:  # if there is documents that we have scraped
        start = time.process_time()
        index = create_and_store_index_in_chroma(documents)
        mid = time.process_time()
        print("Took", mid - start, "seconds to create and store VectorStoreIndex")
    else:   # docs and index r in storage already
        index = load_index_from_chroma()
        # retriever = index.as_retriever(similarity_top_k=5)
        # nodes = retriever.retrieve(
        #     "what book did paul graham write?")
        # print(len(nodes), "nodes were consulted")

    # all texts are turned into vectors and texts are embedded in LLM
    # engine has a retriever and a response synthesizer
    # chat history bounded to leave room for larger context being fed
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_chat_engine(
        streaming=True, similarity_top_k=2, response_mode="refine", chat_mode="context", verbose=True)

    end = time.process_time()
    # query is turned into a vector embedding as well, math ops is carried out by VectorStoreIndex
    # rank all embeddings to find the top k semantically similar ones to query
    while True:
        prompt = input("What do wanna ask me?\n")
        if 'exit' == prompt:
            break
        response = chat_engine.stream_chat(prompt)
        for token in response.response_gen:
            print(token, end="")
        print("\n")

    print("Chat lasted for", time.process_time() - end, "seconds.")


if __name__ == "__main__":
    fresh_scrape = False
    if fresh_scrape:
        documents = get_web_pages()
        create_vector_index(fresh_scrape, documents)
    else:
        create_vector_index(fresh_scrape)
