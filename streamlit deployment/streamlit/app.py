 
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
import tempfile


# App title
st.title("📄 PDF QnA with Gemma2-9b-it (Groq)")

# Sidebar for inputs
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    GROQ_API_KEY = st.text_input("Enter your GROQ API Key:", type="password")

# Upload and Process PDF
if uploaded_file and GROQ_API_KEY:
    with st.spinner("Processing PDF..."):

        # 1. Load the PDF as Documents

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Now pass the file path to PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        # loader = PyPDFLoader(uploaded_file)
        docs = loader.load()  # returns list of Document objects

        # 2. Initialize the chat model
        model = init_chat_model("gemma2-9b-it", model_provider="groq", api_key=GROQ_API_KEY)

        # 3. Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 4. Initialize embeddings and vector store
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embedding_model)
        retriever = vector_store.as_retriever()

        # 5. Define prompt template
        prompt = ChatPromptTemplate(
            input_variables=['context', 'question'],
            messages=[
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=['context', 'question'],
                        template=(
                            "You are an assistant for question-answering tasks. "
                            "Use the following retrieved context to answer the question. "
                            "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
                            "Use three sentences maximum and keep the answer concise.\n\n"
                            "Question: {question}\n"
                            "Context: {context}\n"
                            "Answer:"
                        )
                    )
                )
            ]
        )

        # 6. Function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 7. Define the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        st.success("PDF processed successfully!")

    # Chat Interface
    st.subheader("Ask questions about your PDF 📚")

    # Input user query
    user_query = st.chat_input("Ask a question...")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # If user sends a new message
    if user_query:
        # Save user input
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Thinking... 🤔"):
            response = rag_chain.invoke(user_query)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display all messages in order: user then assistant
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

else:
    st.info("👈 Please upload a PDF and enter your API key to start!")
