import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Document

# OpenAI for embeddings and requests for OpenRouter
import openai
import requests

# Embedded API keys for Streamlit Cloud deployment
OPENROUTER_API_KEY = "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"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")


class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API."""

    model_name: str = "anthropic/claude-3-haiku"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 1000

    def __init__(self, api_key: str, model_name: str = "anthropic/claude-3-haiku", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",  # Streamlit default
            "X-Title": "Conversational RAG Chatbot"
        }

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling OpenRouter API: {str(e)}"


def load_and_process_documents(uploaded_files) -> List[Document]:
    """Load and process uploaded documents."""
    documents = []

    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load document based on file type
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            docs = loader.load()
            documents.extend(docs)

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    return documents


def create_faiss_index_from_documents(documents: List[Document], embeddings) -> Optional[FAISS]:
    """Create FAISS index programmatically from documents."""
    if not documents:
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Split all documents into chunks
    text_chunks = text_splitter.split_documents(documents)

    if not text_chunks:
        return None

    # Create FAISS vector store programmatically
    try:
        vector_store = FAISS.from_documents(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None


def save_faiss_index(vector_store: FAISS, index_path: str = "faiss_index"):
    """Save FAISS index to disk."""
    try:
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        return True
    except Exception as e:
        st.error(f"Error saving FAISS index: {str(e)}")
        return False


def load_faiss_index(embeddings, index_path: str = "faiss_index") -> Optional[FAISS]:
    """Load FAISS index from disk."""
    try:
        if os.path.exists(index_path):
            vector_store = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
    return None


def rebuild_faiss_index(uploaded_files, embeddings):
    """Rebuild FAISS index from uploaded documents."""
    with st.spinner("Processing documents and building FAISS index..."):
        try:
            documents = load_and_process_documents(uploaded_files)

            if not documents:
                st.error("No documents were successfully loaded")
                return None

            vector_store = create_faiss_index_from_documents(documents, embeddings)

            if vector_store:
                if save_faiss_index(vector_store):
                    st.success(f"✅ FAISS index created and saved! Processed {len(documents)} documents.")
                    return vector_store
                else:
                    st.error("Failed to save FAISS index")
            else:
                st.error("Failed to create FAISS index")

        except Exception as e:
            st.error(f"Error rebuilding FAISS index: {str(e)}")

        return None


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "index_stats" not in st.session_state:
        st.session_state.index_stats = {"documents": 0, "chunks": 0}


def main():
    st.set_page_config(
        page_title="Conversational RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Conversational RAG Chatbot")
    st.caption("Powered by LangChain, FAISS, and OpenRouter")  # ✅ FIXED

    # Initialize session state
    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.success("✅ OpenRouter API Key: Configured")

        openai_api_key = st.text_input(
            "OpenAI API Key (for embeddings)",
            type="password",
            value=OPENAI_API_KEY,
            help="Required for document embeddings. You can also set this in Streamlit secrets."
        )

        openrouter_api_key = OPENROUTER_API_KEY

        model_options = [
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet",
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "meta-llama/llama-3-8b-instruct",
            "google/gemma-7b-it"
        ]

        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0
        )

        st.divider()

        st.header("🗄️ FAISS Index Management")

        if st.session_state.faiss_index is not None:
            stats = st.session_state.index_stats
            st.success(f"✅ Index Active: {stats['documents']} docs, {stats['chunks']} chunks")
        else:
            st.info("📄 No index loaded")

        st.subheader("Build New Index")
        uploaded_files = st.file_uploader(
            "Upload documents to build FAISS index",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to create a new FAISS index"
        )

        if st.button("🔨 Build FAISS Index", use_container_width=True) and uploaded_files and openai_api_key:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vector_store = rebuild_faiss_index(uploaded_files, embeddings)

            if vector_store:
                st.session_state.faiss_index = vector_store
                st.session_state.index_stats = {
                    "documents": len(uploaded_files),
                    "chunks": vector_store.index.ntotal
                }

                if openrouter_api_key:
                    llm = OpenRouterLLM(
                        api_key=openrouter_api_key,
                        model_name=selected_model
                    )

                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )

                st.rerun()

        st.subheader("Load Existing Index")
        if st.button("📂 Load Saved Index", use_container_width=True) and openai_api_key:
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vector_store = load_faiss_index(embeddings)

                if vector_store:
                    st.session_state.faiss_index = vector_store
                    st.session_state.index_stats = {
                        "documents": "Unknown",
                        "chunks": vector_store.index.ntotal
                    }

                    if openrouter_api_key:
                        llm = OpenRouterLLM(
                            api_key=openrouter_api_key,
                            model_name=selected_model
                        )

                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                            return_source_documents=True
                        )

                    st.success("✅ FAISS index loaded successfully!")
                    st.rerun()
                else:
                    st.warning("⚠️ No saved FAISS index found")
            except Exception as e:
                st.error(f"Error loading FAISS index: {str(e)}")

        if st.button("🗑️ Clear Index", use_container_width=True):
            st.session_state.faiss_index = None
            st.session_state.qa_chain = None
            st.session_state.index_stats = {"documents": 0, "chunks": 0}
            st.success("Index cleared!")
            st.rerun()

        st.divider()

        if st.button("💬 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.header("💬 Chat Interface")

    if not openrouter_api_key:
        st.error("OpenRouter API key is not configured. Please check the app configuration.")
        return

    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key for embeddings in the sidebar, or configure it in Streamlit secrets.")
        st.info("💡 **Quick Start**: You can use a demo OpenAI key for testing. Get one at: https://platform.openai.com/api-keys")
        return

    if st.session_state.qa_chain is None:
        st.info("📋 **Getting Started:**\n"
                "1. Enter your OpenAI API key in the sidebar\n"
                "2. Upload PDF or TXT documents\n"
                "3. Click 'Build FAISS Index' to process documents\n"
                "4. Start chatting with your documents!")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(source[:500] + "..." if len(source) > 500 else source)
                        st.divider()

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    sources = [doc.page_content for doc in source_docs]

                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources):
                                st.write(f"**Source {i+1}:**")
                                st.write(source[:500] + "..." if len(source) > 500 else source)
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })


if __name__ == "__main__":
    main()
