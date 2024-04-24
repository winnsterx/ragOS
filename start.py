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
from agent import Agent


def load_index_from_chroma(db_name, collection_name):
    db = chromadb.PersistentClient(path="./"+db_name)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context)
    return index


def create_and_store_index_in_chroma(documents, db_name, collection_name):
    db = chromadb.PersistentClient(path="./"+db_name)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # this loads it and saves it to storage context
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context)

    return vector_index


def create_vector_index(fresh_scrape, db_name, documents=[]):
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    Settings.llm = Ollama(model="llama2", request_timeout=100)
    index = None

    if fresh_scrape:  # if there is documents that we have scraped
        start = time.process_time()
        index = create_and_store_index_in_chroma(
            documents, db_name, "paul_graham_wiki")
        mid = time.process_time()
        print("Took", mid - start, "seconds to create and store VectorStoreIndex")
    else:   # docs and index r in storage already
        index = load_index_from_chroma(db_name)

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_chat_engine(
        streaming=True, similarity_top_k=2, response_mode="tree_summarize", chat_mode="context", verbose=True)

    end = time.process_time()
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
    os = Agent()
    # url1 = "https://en.wikipedia.org/wiki/Paul_Graham_(programmer)"
    # url2 = "https://en.wikipedia.org/wiki/Awkwafina"
    # url3 = "https://en.wikipedia.org/wiki/Vin_Diesel"
    # doc1 = os.scrape_web_pages(url1)
    # doc2 = os.scrape_web_pages(url2)
    # doc3 = os.scrape_web_pages(url3)
    # os.create_collection(
    #     "paul_graham", documents=doc1, description="Use this to get overview of Paul Graham's life")
    # os.create_collection(
    #     "awkwafina", documents=doc2, description="Use this to get overview of Awkwafina's life")
    # os.create_collection(
    #     "vin_diesel", documents=doc3, description="Use this to get overview of Vin Diesel's life")
    os.start()
    # fresh_scrape = True
    # print(db.list_collections())

    # if fresh_scrape:

    #     create_vector_index(fresh_scrape, db_name, documents)
    # else:
    #     create_vector_index(fresh_scrape, db_name)
