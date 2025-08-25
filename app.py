import streamlit as st
import os
import tempfile
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Document
import requests

# 🔑 Embedded API Key (your OpenRouter key, hard-coded)
OPENROUTER_API_KEY = "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"

# ----------------------
# Custom OpenRouter LLM
# ----------------------
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
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


# ----------------------
# OpenRouter Embeddings
# ----------------------
class OpenRouterEmbeddings:
    """Embedding wrapper using OpenRouter API."""

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]):
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str):
        return self._embed(text)

    def _embed(self, text: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": text
        }
        response = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


# ----------------------
# Document Processing
# ----------------------
def load_and_process_documents(uploaded_files) -> List[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
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
            os.unlink(tmp_file_path)

    return documents


def create_faiss_index_from_documents(documents: List[Document], embeddings) -> Optional[FAISS]:
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_chunks = text_splitter.split_documents(documents)
    if not text_chunks:
        return None

    try:
        vector_store = FAISS.from_documents(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None


# ----------------------
# Streamlit App
# ----------------------
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None


def main():
    st.set_page_config(
        page_title="Conversational RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Conversational RAG Chatbot")
    st.caption("Powered by LangChain, FAISS, and OpenRouter")

    initialize_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.success("✅ Using embedded OpenRouter API Key")

        model_options = [
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet",
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "meta-llama/llama-3-8b-instruct",
            "google/gemma-7b-it"
        ]
        selected_model = st.selectbox("Select Model", model_options, index=0)

        st.divider()

        st.subheader("Build FAISS Index")
        uploaded_files = st.file_uploader(
            "Upload PDF/TXT documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )

        if st.button("🔨 Build FAISS Index", use_container_width=True) and uploaded_files:
            embeddings = OpenRouterEmbeddings(api_key=OPENROUTER_API_KEY)
            documents = load_and_process_documents(uploaded_files)
            vector_store = create_faiss_index_from_documents(documents, embeddings)

            if vector_store:
                st.session_state.faiss_index = vector_store
                llm = OpenRouterLLM(api_key=OPENROUTER_API_KEY, model_name=selected_model)
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                st.success("✅ FAISS index built successfully!")

    st.header("💬 Chat Interface")
    if st.session_state.qa_chain is None:
        st.info("📋 Please upload documents and build the FAISS index first.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    answer = result["result"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
