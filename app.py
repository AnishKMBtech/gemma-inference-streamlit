import streamlit as st
import threading
import queue
import time
import os
from llama_cpp import Llama

# Page configuration
st.set_page_config(
    page_title="Gemma Chat",
    page_icon="🤖",
    layout="centered"
)

# Path to your GGUF model (must be downloaded locally)
MODEL_PATH = "gemma-3-270m-it-UD-IQ2_M.gguf"

@st.cache_resource
def load_model():
    """Load GGUF model with llama-cpp (optimized for 2 cores / 2GB RAM)"""
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,         # keep context smaller
            n_threads=2,        # limit to 2 CPU cores
            n_batch=64,         # small batch to save RAM
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

def format_prompt(message, history):
    """Format prompt for chat"""
    conversation = ""
    for msg in history[-6:]:
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        else:
            conversation += f"Assistant: {msg['content']}\n"
    conversation += f"User: {message}\nAssistant:"
    return conversation

def generate_response_threaded(llm, prompt, result_queue, max_tokens=256, temp=0.7):
    """Generate response in a thread with llama-cpp"""
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["User:", "Assistant:"]
        )
        response = output["choices"][0]["text"].strip()
        result_queue.put(("success", response))
    except Exception as e:
        result_queue.put(("error", str(e)))

def main():
    st.title("🤖 Gemma Chat (GGUF - Low Resource)")

    # Load model
    if "llm" not in st.session_state:
        with st.spinner("Loading Gemma-3-270M IT GGUF model..."):
            llm = load_model()
            if llm is not None:
                st.session_state.llm = llm
            else:
                st.stop()

    # Controls
    col1, col2 = st.columns([3, 1])
    with col2:
        max_tokens = st.slider("Max tokens", 50, 400, 200)
        temp = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        if st.button("Clear", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Init messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.write("Thinking...")

            result_queue = queue.Queue()
            formatted_prompt = format_prompt(prompt, st.session_state.messages[:-1])

            thread = threading.Thread(
                target=generate_response_threaded,
                args=(st.session_state.llm, formatted_prompt, result_queue, max_tokens, temp)
            )
            thread.start()

            start_time = time.time()
            response = ""

            while thread.is_alive():
                if time.time() - start_time > 25:  # smaller timeout
                    response = "Response timeout. Please try again."
                    break
                time.sleep(0.1)

            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success":
                    response = result
                else:
                    response = f"Error: {result}"

            thread.join(timeout=1)

            if not response.strip():
                response = "I apologize, I couldn't generate a proper response."

            response_placeholder.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
