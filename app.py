from transformers import AutoModel, AutoTokenizer
import streamlit as st
from PIL import Image
import re
import os
import uuid

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the model and tokenizer only once
@st.cache_resource
def load_model(model_name):
    if model_name == "OCR for English or Hindi (CPU)":
        tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
        model = AutoModel.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id).eval().to('cuda')
    return model, tokenizer

if "model" not in st.session_state or "tokenizer" not in st.session_state:
    model, tokenizer = load_model("OCR for English or Hindi (CPU)")
    st.session_state.update({"model": model, "tokenizer": tokenizer})

# Function to run the GOT model for multilingual OCR
def run_ocr(image, model, tokenizer):
    image_path = f"{uuid.uuid4()}.png"
    image.save(image_path)

    try:
        res = model.chat(tokenizer, image_path, ocr_type='ocr')
        return res if isinstance(res, str) else str(res)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        os.remove(image_path)

# Function to highlight keyword in text
def highlight_text(text, search_term):
    return re.sub(re.escape(search_term), lambda m: f'<span style="background-color: red;">{m.group()}</span>', text, flags=re.IGNORECASE) if search_term else text

# Streamlit App
st.title(":blue[Optical Character Recognition Application]")
st.write("upload image for ocr")

# Create two columns
col1, col2 = st.columns(2)

# Left column - Display the uploaded image
with col1:
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

# Right column - Model selection, options, and displaying extracted text
with col2:
    model_option = st.selectbox("Select Model", ["OCR for English or Hindi (CPU)", "OCR for English (GPU)"])
    
    if st.button("Run OCR"):
        if uploaded_image:
            with st.spinner("Processing..."):
                model, tokenizer = load_model(model_option)
                result_text = run_ocr(image, model, tokenizer)
                if "Error" not in result_text:
                    st.session_state["extracted_text"] = result_text
                else:
                    st.error(result_text)
        else:
            st.error("Please upload an image before running OCR.")

    # Display the extracted text if it exists in session state
    if "extracted_text" in st.session_state:
        search_term = st.text_input("Enter a word or phrase to highlight:")
        st.subheader("Extracted Text:")
        st.markdown(f'<div style="white-space: pre-wrap;">{highlight_text(st.session_state["extracted_text"], search_term)}</div>', unsafe_allow_html=True)
