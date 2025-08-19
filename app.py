import streamlit as st
import threading
import queue
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Set environment variable to disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Page configuration
st.set_page_config(
    page_title="Gemma Chat",
    page_icon="🤖",
    layout="centered"
)

# Model configuration
MODEL_NAME = "google/gemma-3-270m"

@st.cache_resource
def load_model_pipeline():
    """Load the model pipeline with optimized settings"""
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimal settings
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            device_map=None,  # Keep on CPU for stability
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Force CPU
            torch_dtype=torch.float32
        )
        
        return pipe
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

def format_prompt(message, history):
    """Format prompt for Gemma"""
    conversation = ""
    # Only keep last 3 exchanges for context
    for msg in history[-6:]:  
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        else:
            conversation += f"Assistant: {msg['content']}\n"
    
    conversation += f"User: {message}\nAssistant:"
    return conversation

def generate_response_threaded(pipe, prompt, result_queue, max_tokens=256, temp=0.7):
    """Generate response in a separate thread"""
    try:
        response = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
            truncation=True
        )
        
        text = response[0]["generated_text"].strip()
        result_queue.put(("success", text))
        
    except Exception as e:
        result_queue.put(("error", str(e)))

def main():
    st.title("🤖 Gemma Chat")
    
    # Load model
    if "pipeline" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.pipeline = load_model_pipeline()
    
    if st.session_state.pipeline is None:
        st.error("Failed to load model")
        st.stop()
    
    # Simple controls
    col1, col2 = st.columns([3, 1])
    with col2:
        max_tokens = st.slider("Max tokens", 50, 500, 256)
        temp = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        if st.button("Clear", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response with threading
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Show loading
            response_placeholder.write("Thinking...")
            
            # Create queue for thread communication
            result_queue = queue.Queue()
            
            # Format prompt
            formatted_prompt = format_prompt(prompt, st.session_state.messages[:-1])
            
            # Start generation in thread
            thread = threading.Thread(
                target=generate_response_threaded,
                args=(st.session_state.pipeline, formatted_prompt, result_queue, max_tokens, temp)
            )
            thread.start()
            
            # Wait for result with timeout
            start_time = time.time()
            response = ""
            
            while thread.is_alive():
                if time.time() - start_time > 30:  # 30 second timeout
                    response = "Response timeout. Please try again."
                    break
                time.sleep(0.1)
            
            # Get result from queue
            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success":
                    response = result
                else:
                    response = f"Error: {result}"
            
            thread.join(timeout=1)
            
            # Clean up response
            if response and not response.startswith("Error") and not response.startswith("Response timeout"):
                # Remove common artifacts
                response = response.replace("User:", "").replace("Assistant:", "")
                response = response.strip()
                
                if not response:
                    response = "I apologize, but I couldn't generate a proper response."
            
            # Display final response
            response_placeholder.write(response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
