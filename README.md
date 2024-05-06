- add LLM based rerankers to semantic retrieval pipeline to increase nodes accuracies
- add live website crawling features (extract text and feed in as context)
- context window size rn: how is LLM retaining context rn? through 
- integrat with agents (https://medium.com/llamaindex-blog/data-agents-eed797d7972f) to perform tasks using tools (going beyond reading context and writing answers)
  - or to read from external searchers (extract info from API and add that into context for search)
- fine-tune model on the fly to increase accuracy to my personalization? 
  - like the model gets better at answering questions about ME?! is this possible? 

  
To run:
1. activate virtual env with source: .venv/bin/activate
2. install dependencies: pip3 install -r requirements.txt
3. start chat engine: python3 start.py

To run: 
1. start streamlit app: `python3 -m streamlit run home.py  `


To do:
- [ ] add Notion ingestion pipeline (can this be in real-time)
- [ ] add notes app ingestion pipeline (real-time!)
- [ ] add LLM based reranker to improve nodes selection
- [ ] add file upload knowledge upgrade path
- [ ] add ability to update or remove exisiting RAGs collection 
- [x] fix bug that crashes the app when retriever fails to get notes
  - [x] make it fall back on LLM knowledge when json returns nil
- [x] fix timeout bug in LLM selector: LLM client timing out on request
  - [x] ollama needed an upgrade
- [ ] tailor retrieval mechanism (https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/retrievers/)  
- [ ] router is not very good atm. it gets routed all over the place.
- [ ] 

v0: personalized RAG with local LLM that is actually useful
- [ ] use personal data like notion, slack etc 
- [ ] use basic pipelines like websites, txt docs, etc
- [ ] basic RAG retrieval
  - [ ] multiple collections vs larger individual collections
  - [ ] why is the score so low? 
v1: improved retrieval accuracy and speed
- [ ] create testing module
  - [ ] setup testing standards (with same local info, prompts) to track performance
- [ ] support all file types 
- [ ] reduce memory requirement 
  - [ ] docstore + index_store in storage_context both use memory, not persisting to disk. could this cause the excessive memory requirements? but still gotta load it into memory tho...? 
v2: additional pipelines and data integrations 