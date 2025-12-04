import streamlit as st
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import numpy as np
import io
import requests

# Page configuration
st.set_page_config(
    page_title="Clothes Segmentation",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define clothing labels
CLOTHING_LABELS = [1, 4, 5, 6, 7, 8, 9, 10, 16, 17]

# Label mapping
id2label = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm",
    15: "Right-arm", 16: "Bag", 17: "Scarf"
}

@st.cache_resource
def load_model(model_path):
    """Load model and processor (cached for performance)"""
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_path)
    return processor, model

def get_clothing_mask(image, processor, model):
    """Process image and return segmentation results"""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    clothing_mask = np.zeros_like(pred_seg.numpy())
    for label in CLOTHING_LABELS:
        clothing_mask[pred_seg == label] = label
    
    image_array = np.array(image)
    output_image = image_array.copy()
    output_image[clothing_mask == 0] = [255, 255, 255]
    
    return pred_seg, output_image

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        return image, None
    except Exception as e:
        return None, f"Error loading image from URL: {str(e)}"

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    body {
        background-color: #2c2c2c;
        color: #f5f5f5;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(to right, #2196f3, #21cbf3);
        color: #f5f5f5;
        font-weight: 600;
        font-family: 'Roboto', sans-serif;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #1976d2, #1e88e5);
        color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .image-container {
        background-color: #333;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 10px;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    .image-title {
        font-size: 18px;
        font-weight: bold;
        color: #e0e0e0;
        margin-bottom: 15px;
    }
    img {
        border-radius: 10px;
        max-height: 500px;
        object-fit: contain;
    }
    .upload-section {
        background-color: #333;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 36px; color: #f5f5f5; margin-bottom: 5px;">👔 Clothes Segmentation System</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Clothes Segmentation")

with st.sidebar:
    with st.expander("About"):
        st.markdown("""
        This Clothes Segmentation System extracts clothing items from images.
        
        - Built using SegFormer-B2 Model
        - Detects 18 different categories
        - Supports hats, clothes, shoes, bags, and more
        """)

    with st.expander("How to Use"):
        st.markdown("""
        1. Choose input method (Upload or URL)
        2. Provide an image with a person
        3. Click 'Segment Clothes'
        4. View the original and segmented results side by side
        """)

    with st.expander("Detected Categories"):
        st.markdown("""
        - **Clothes**: Upper-clothes, Pants, Skirt, Dress
        - **Accessories**: Hat, Belt, Bag, Scarf
        - **Footwear**: Left-shoe, Right-shoe
        """)

    st.markdown("---")

# Load model
model_path = "assets/weights/"
try:
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.processor, st.session_state.model = load_model(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"⚠️ Could not load model from '{model_path}'")

# Main content
# Toggle between upload and URL
input_method = st.radio(
    "Choose input method:",
    ["Upload Image", "Image URL"],
    horizontal=True,
    label_visibility="collapsed"
)

image = None
error_message = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image to segment",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a person wearing clothes",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    image_url = st.text_input(
        "Enter image URL",
        placeholder="https://example.com/image.jpg",
        label_visibility="collapsed"
    )
    if image_url:
        image, error_message = load_image_from_url(image_url)
        if error_message:
            st.error(error_message)

if image is not None and model_loaded:
    # Create two columns for layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Display uploaded image on the left
        st.markdown(
            """
            <div class="image-container">
                <div class="image-title">📤 Uploaded Image</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        
        # Segment button below the image
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button('🚀 Segment Clothes', use_container_width=True):
            st.session_state.process_image = True
    
    with col_right:
        # Process and display segmented image if button was clicked
        if 'process_image' in st.session_state and st.session_state.process_image:
            with st.spinner("Processing image... This may take a moment"):
                try:
                    pred_seg, clothing_only = get_clothing_mask(
                        image,
                        st.session_state.processor,
                        st.session_state.model
                    )
                    
                    # Display segmented clothes on the right
                    st.markdown(
                        """
                        <div class="image-container">
                            <div class="image-title">👕 Segmented Clothes</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(clothing_only, use_container_width=True)
                    
                    # Download button
                    st.markdown("<br>", unsafe_allow_html=True)
                    img_clothes = Image.fromarray(clothing_only.astype(np.uint8))
                    buf_clothes = io.BytesIO()
                    img_clothes.save(buf_clothes, format='PNG')
                    
                    st.download_button(
                        label="⬇️ Download Segmented Image",
                        data=buf_clothes.getvalue(),
                        file_name="clothes_segmented.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # Reset the flag
                    st.session_state.process_image = False
                    
                except Exception as e:
                    st.error(f"❌ Error during segmentation: {str(e)}")
                    st.session_state.process_image = False

elif image is None and model_loaded and not error_message:
    st.info("👆 Please upload an image or enter an image URL to get started")
elif not model_loaded:
    st.warning("⚠️ Model not loaded. Please check the model path.")

# Footer
st.markdown("""<hr style="margin-top: 50px;"/>""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; padding: 10px; font-size: 14px;">
        Made by <strong>Abdelrahman Tawfik</strong>
    </div>
    """,
    unsafe_allow_html=True
)