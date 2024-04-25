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
- [ ] fix bug that crashes the app when retriever fails to get notes
  - [ ] make it fall back on LLM knowledge when json returns nil
- [x] fix timeout bug in LLM selector: LLM client timing out on request
  - [x] ollama needed an upgrade
- [ ] tailor retrieval mechanism (https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/retrievers/)  
- [ ] router is not very good atm. it gets routed all over the place.

Traceback (most recent call last):
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 584, in _run_script
    exec(code, module.__dict__)
  File "/Users/winniex/code/ragOS/home.py", line 58, in <module>
    response = st.session_state["agent"].chat_engine.chat(prompt)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/callbacks/utils.py", line 41, in wrapper
    return func(self, *args, **kwargs)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/chat_engine/context.py", line 162, in chat
    context_str_template, nodes = self._generate_context(message)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/chat_engine/context.py", line 109, in _generate_context
    nodes = self._retriever.retrieve(message)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/instrumentation/dispatcher.py", line 274, in wrapper
    result = func(*args, **kwargs)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/base/base_retriever.py", line 244, in retrieve
    nodes = self._retrieve(query_bundle)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/retrievers/router_retriever.py", line 92, in _retrieve
    result = self._selector.select(self._metadatas, query_bundle)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/base/base_selector.py", line 87, in select
    return self._select(choices=metadatas, query=query_bundle)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/selectors/llm_selectors.py", line 215, in _select
    parsed = self._prompt.output_parser.parse(prediction)
  File "/Users/winniex/Library/Python/3.9/lib/python/site-packages/llama_index/core/output_parsers/selection.py", line 97, in parse
    raise ValueError(f"Failed to convert output to JSON: {output!r}")
ValueError: Failed to convert output to JSON: "A clever question!\n\nSince the choices don't seem to have any relevance to retrievers, I'll return an empty list.\n\nHere's the JSON output:\n```\n[]\n```"