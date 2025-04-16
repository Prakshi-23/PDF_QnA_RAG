# 📄 PDF\_QnA\_RAG

PDF\_QnA\_RAG is an intelligent question-answering system that allows users to upload a PDF file and interact with its content using Retrieval-Augmented Generation (RAG). It combines document chunking, semantic search, and large language model responses to generate accurate, context-aware answers.

## 🚀 Features

- Upload and read content from PDF files
- Split large documents into manageable text chunks
- Generate embeddings using `sentence-transformers`
- Store and search embeddings using FAISS
- Retrieve relevant context for any user question
- Generate answers using `gemma2-9b-it` via the `groq` API

## 🧠 How It Works

1. **PDF Processing:** Text is extracted and chunked from the uploaded PDF.
2. **Embedding Generation:** Each chunk is embedded using a pretrained transformer model.
3. **Vector Store:** Embeddings are stored in a FAISS index for fast retrieval.
4. **Question Handling:** When a user asks a question, similar chunks are retrieved.
5. **Answer Generation:** The question and context are passed to a language model for answering.

## 🛠 Tech Stack

- Python
- PyPDFLoader (for PDF parsing)
- `sentence-transformers` (for embeddings)
- `faiss` (for vector search)
- `groq` API + `langchain` (for LLM-based answering)
- `gemma-2b-it` model

## 📈 Example Use Case

> Upload a financial report PDF and ask:\
> *"What was the net income in Q4?"*\
> The system retrieves the relevant data from the document and provides an accurate, LLM-generated answer.



---

Made with ❤️ using LangChain, FAISS, and Groq.

