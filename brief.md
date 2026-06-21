Technical Assessment — Generative AI Expert

Problem Statement
You are tasked with building a RAG (Retrieval-Augmented Generation) chatbot that allows a user to interact with the content of a provided PDF document through a conversational interface.
The chatbot must answer questions strictly based on the content of the document. If the answer cannot be found in the document, the chatbot must explicitly say so, it should never fabricate information or answer from general knowledge.
Every response must include references to the source page/section that were used to generate the answer, so the user can trace back the information to the original document.
The solution must be built using LangChain / LangGraph as the orchestration layer, a cloud-hosted vector database for retrieval, and exposed through a Chainlit or Streamlit user interface. The application must run locally via a single Docker container.
You will be evaluated not only on whether the solution works, but on how you think: your commit history, your design decisions around chunking and retrieval, your choice of models, and how you handle configuration and secrets. A working solution with poor engineering practices will score lower than a thoughtful, well-structured solution.
Deliverables
1. GitHub Repository
A public GitHub repository containing the full source code. The commit history must reflect your development process — we expect to see incremental, meaningful commits, not a single initial commit with the entire solution. Commit messages should be clear and descriptive. The repository must contain a complete, runnable application including:
•	The ingestion script used to process the PDF and populate the vector database
•	The RAG pipeline built with LangChain / LangGraph
•	The conversational UI built with Chainlit or Streamlit
•	A Dockerfile to build and run the application locally

2. Source Code
In addition to the GitHub repository, submit a ZIP file of the complete source code separately.

3. Configuration
•	A .env.example file listing all required environment variables with placeholder values and a short description for each
•	The .env file itself must never appear in the repository or its git history
•	Your own API keys must never be committed
Technical Requirements
Stack
•	Orchestration: LangChain / LangGraph
•	UI: Chainlit (preferred) or Streamlit
•	Vector database: a cloud-hosted vector store with a free tier (e.g. Pinecone, Qdrant Cloud) — you ingest the PDF once before submitting; the index must be ready to be queried by the chatbot
•	Embedding model: free to use any model, open source or proprietary — document the model name and version used
•	Chat model: free to use any model, open source or proprietary — document the model name and version used

Chatbot behaviour
•	Answers must be grounded strictly in the content of the provided PDF
•	If the answer is not in the document, the chatbot must say so explicitly
•	Every response MUST display the page number and/or section from the PDF where the answer was found — this is mandatory and will be tested to validate the answer against the source document
•	The chatbot must support multi-turn conversation within a session

What will be assessed beyond the running application
Even though we will not run the ingestion step, the following will be carefully reviewed during the technical interview:
•	Chunking strategy — why you chose a specific chunk size, overlap, and splitting method for this document type
•	Ingestion pipeline — how you loaded, processed, and embedded the PDF before populating the vector store
•	Vector store fields — what metadata you stored alongside each chunk (e.g. page number, section, source) and why
•	Embedding model choice — which model you used, why, and how it affects retrieval quality
Be prepared to walk through your ingestion script and justify every decision. The quality of your reasoning matters as much as the quality of your code.
We will run your application using ONLY these steps:
1.	git clone <your-repo>
2.	Fill in the .env file with the required API keys using your .env.example
3.	docker build -t chatbot:1.0 .
4.	docker run -p <PORT>:<PORT> --env-file .env chatbot:1.0
5.	Open localhost:<PORT> in a browser and start chatting

The port your application runs on is your choice, document it clearly in your README so we can do correct port mapping. Since the vector index is already populated in the cloud, there is no ingestion step to run. The application must be ready to answer questions immediately after the container starts. If the setup requires any additional steps beyond the above, document it in your README file.
