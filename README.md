
# Retrieval-Augmented Generation (RAG) Pipeline

## Overview
This repository provides for a Retrieval-Augmented Generation (RAG) system, which integrates retrieval-based approaches with generative models.
The system first retrieves relevant documents or information from a knowledge base and then uses a generative model to synthesize a coherent response based on the retrieved information.

## Steps Involved in RAG
1. Data Ingestion
2. Indexing & Storing
3. Retrieval
4. Response Synthesis
5. Query/Chat Engine


## 1. Data Ingestion
The Data Ingestion phase involves collecting, processing, and preparing the data for storage and retrieval.
It can handle structured and unstructured data sources such as text documents, databases, APIs, or web scraping.
### Features
Ingests data from multiple formats (PDFs, text files, JSON, etc.).
Data validation and preprocessing steps to ensure quality.
Supports both batch and real-time ingestion.
### Key Tools/Technologies
- **llamaIndex**: Used for loading and reading documents from files


                     from llamaIndex import SimpleDirectoryReader
                     ###Load data from PDF files or directories
                     documents = SimpleDirectoryReader(input_files=['data/transformers.pdf']).load_data()
-----------------------------------------------------
## 2. Indexing & Storing
### Why Indexing?
- Quick Retrieval: Speeding up the process of finding relevant information.
- Enhanced Accuracy: Improves the relevance and quality of information retrieved.
- Scalability: Allows the system to efficiently handle large data volumes.

### Features
- Creates dense embeddings for documents using a custom embedding model.
- Supports various pre-trained models for embedding generation.
- Efficient embedding handling for large document sets, enabling fast retrieval.

### Key Tools/Technologies
- Custom Embedding Model: Define a custom embedding model using SentenceTransformer for embedding generation. 
- llamaIndex: Utilized to create vector-based indexes for document retrieval.
  
                     from llama_index.core import VectorStoreIndex
                     ###Create a vector-based index from documents
                     index = VectorStoreIndex.from_documents(documents,embed_model=embed_model)
********************************************
## 3. Retrieval
The Retrieval component is responsible for fetching the most relevant documents based on a user query.
This step is crucial for the RAG system since the quality of retrieved documents directly affects the generated response.
### Features
Supports semantic search using embeddings.
Ranked retrieval using relevance scoring algorithms.
### Key Tools/Technologies
- **llamaIndex**: Used to convert the document index into a retriever for querying.For example:

                      retriever = index.as_retriever()
                      ###Retrieve documents based on a query
                      query = "What is self attention?"
                      results = retriever.retrieve(query)

********************************************
## 4. Response Synthesis
In the Response Synthesis phase, a generative model such as a large language model (LLM) processes the retrieved documents and generates a coherent,
contextually relevant response.
### Features
Retrieval-augmented generation that combines relevant documents with the power of generative model.
Combines retrieved documents to produce detailed answers.
Supports various response formats, including conversational and structured outputs.
### Key Tools/Technologies
- **Hugging Face Transformers**:for accessing and utilizing pre-trained language models.
- **llamaIndex**: For managing the retrieval of documents and preparing them for the response synthesis.
*Response Synthesizer:* A function that integrates the LLM for generating responses based on the retrieved data. For example:

                          response_synthesizer = get_response_synthesizer(llm=llm)

*******************************************  
## 5. Query/Chat Engine 
The Query/Chat Engine is the user-facing component that takes queries as input, retrieves relevant information,
and returns a synthesized response. It also supports conversational interactions, maintaining context across multiple turns.
*******************************************


## Installation
## Prerequisites
Python 3.8+
LlamaIndex library
Transformers library for LLaMA model integration.
*****************************************************************************************************************************************
************************************************************************************************************************
# Contextual Retrieval-Augmented Generation (RAG)

## Overview
A Contextual RAG system retrieves relevant pieces of information from a dataset or knowledge base and uses this context to generate more informed and accurate responses. This implementation uses:

- Hugging Face Transformers: open-source LLM.

- LangChain: For orchestrating the RAG pipeline, including document loading, retrieval, and contextual generation.

- Chroma: For efficient vector-based retrieval.
---------------------------------------------------------
## Features

- Retrieval of context-specific documents using Chroma.

- Integration with Hugging Face's free LLM for cost-effective deployment.

- Easily configurable to your dataset or domain.
 ----------------------------------------------------------------
 ## Installation
 - Install Dependencies:Ensure Python 3.8+ is installed.
 - Install  chromadb

              pip install chromadb
 - Download a Free LLM from Hugging Face:

              from transformers import AutoModelForCausalLM, AutoTokenizer
              model_name = "meta-llama/Meta-Llama-3.1-8B"
              model = AutoModelForCausalLM.from_pretrained(model_name)
              tokenizer = AutoTokenizer.from_pretrained(model_name)
   -----------------------------------------------------------------------
  ## How It Works

1. Document Preprocessing: The dataset is preprocessed and embedded using a sentence transformer model.

2. Chroma Indexing: Document embeddings are indexed for fast similarity search.

3. Retrieval: When a query is input, the system retrieves the most relevant documents.

4. Contextual Generation: The retrieved context is appended to the query and passed to the LLM for generation.
#### To use the Contextual Compression Retriever, you'll need:
- a base retriever.
- a Document Compressor.
 The Contextual Compression Retriever passes queries to the base retriever, takes the initial documents and passes them through the 
 Document Compressor. The Document Compressor takes a list of documents and shortens it by reducing the contents of documents or 
 dropping documents altogether.

         compressor = LLMChainExtractor.from_llm(llm)
         compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=retriever
         )
         compressed_docs = compression_retriever.invoke("How do ViGGO's 'List' slots differ from other NLG datasets like E2E and Hotel?")
         print(compressed_docs)
***********************************************************************************************************************************
