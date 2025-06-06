
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
```python

                     from llamaIndex import SimpleDirectoryReader
                     ###Load data from PDF files or directories
                    documents = SimpleDirectoryReader(input_files=['data/transformers.pdf']).load_data()
```
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
  ```python
                     from llama_index.core import VectorStoreIndex
                     ###Create a vector-based index from documents
                     index = VectorStoreIndex.from_documents(documents,embed_model=embed_model)
  ```
********************************************
## 3. Retrieval
The Retrieval component is responsible for fetching the most relevant documents based on a user query.
This step is crucial for the RAG system since the quality of retrieved documents directly affects the generated response.
### Features
Supports semantic search using embeddings.
Ranked retrieval using relevance scoring algorithms.
### Key Tools/Technologies
- **llamaIndex**: Used to convert the document index into a retriever for querying.For example:
```python
                      retriever = index.as_retriever()
                      ###Retrieve documents based on a query
                      query = "What is self attention?"
                      results = retriever.retrieve(query)

```

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
```python
                          response_synthesizer = get_response_synthesizer(llm=llm)
```
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
```python
              from transformers import AutoModelForCausalLM, AutoTokenizer
              model_name = "meta-llama/Meta-Llama-3.1-8B"
              model = AutoModelForCausalLM.from_pretrained(model_name)
              tokenizer = AutoTokenizer.from_pretrained(model_name)
```
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
```python
         compressor = LLMChainExtractor.from_llm(llm)
         compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=retriever
         )
         compressed_docs = compression_retriever.invoke("How do ViGGO's 'List' slots differ from other NLG datasets like E2E and Hotel?")
         print(compressed_docs)
```
***********************************************************************************************************************************
***********************************************************************************************************************************
# Parent Document Retriever
![image](https://github.com/user-attachments/assets/d1f1df77-f6fa-4fbb-ba6a-6f4fc1f3d52e)

## How to use the Parent Document Retriever
When splitting documents for retrieval, there are often conflicting desires:

- You may want to have small documents, so that their embeddings can most accurately reflect their meaning. If too long, then the embeddings can lose meaning.
- You want to have long enough documents that the context of each chunk is retained.
The ParentDocumentRetriever strikes that balance by splitting and storing small chunks of data. During retrieval, it first fetches the small chunks but then looks up the parent ids for those chunks and returns those larger documents.

Note that "parent document" refers to the document that a small chunk originated from. This can either be the whole raw document OR a larger chunk.
```python
          
          from langchain.retrievers import ParentDocumentRetriever
          from langchain.storage import InMemoryStore
          from langchain_chroma import Chroma
          from langchain_text_splitters import RecursiveCharacterTextSplitter

```     
API Reference:[InMemoryStore](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryStore.html) | [TextLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.text.TextLoader.html) | [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

***************************************************************************************************************
***************************************************************************************************************
