- add live website crawling features (extract text and feed in as context)
- context window size rn: how is LLM retaining context rn? through 
- integrat with agents (https://medium.com/llamaindex-blog/data-agents-eed797d7972f) to perform tasks using tools (going beyond reading context and writing answers)
  - or to read from external searchers (extract info from API and add that into context for search)
- fine-tune model on the fly to increase accuracy to my personalization? 
  - like the model gets better at answering questions about ME?! is this possible? 
- add LLM based rerankers to semantic retrieval pipeline to increase nodes accuracies
  
To run:
1. activate virtual env with source: .venv/bin/activate
2. install dependencies: pip3 install -r requirements.txt
3. start chat engine: python3 start.py

