import streamlit as st
import threading
import queue
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
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
MODEL_NAME = "ANISH-j/gemma"

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer directly"""
    try:
        # Load model directly as specified
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

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

def generate_response_threaded(model, tokenizer, prompt, result_queue, max_tokens=256, temp=0.7):
    """Generate response in a separate thread using direct model inference"""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (remove the input prompt)
        response = generated_text[len(prompt):].strip()
        
        result_queue.put(("success", response))
        
    except Exception as e:
        result_queue.put(("error", str(e)))

def main():
    st.title("🤖 Gemma Chat")
    
    # Load model
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        with st.spinner("Loading ANISH-j/gemma model..."):
            model, tokenizer = load_model_and_tokenizer()
            if model is not None and tokenizer is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
            else:
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
                args=(st.session_state.model, st.session_state.tokenizer, formatted_prompt, result_queue, max_tokens, temp)
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
