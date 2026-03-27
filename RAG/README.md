Implemented a from-scratch RAG pipeline using Ollama where I built my own in-memory vector store (manual embedding + cosine similarity). 

Initially  `script.py` file can read the cat facts and state it back.. But each time the chunks get loaded from the beginning... 
Hence `index.py` and `ask.py` were given birth.. 
Now I wanted the same model to ingest data from diffrent files and answer about them together compare and contrast and all.. so `multi-index.py` and `multi-ask.py` 
Then worked on "cats and dogs" & "dogs and cats" meaning the same thing.. ​

First it was over a single document and then extended it to multi-document, label-aware retrieval across cat and dog knowledge bases. 

Designed interactive query scripts (ask.py, multi-ask.py) that retrieve relevant chunks based on semantic similarity and thresholds, construct context-constrained prompts, and stream answers from a local Llama model. 

Documented a roadmap for upgrading retrieval quality (hybrid search, re-ranking, improved chunking), adding evaluation metrics, and moving from tutorial-level RAG to production-grade systems.
