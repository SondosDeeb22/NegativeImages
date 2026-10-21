import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Image Negative Transform",
    layout="centered",
)

# Custom CSS for lighter theme and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stApp {
        background: linear-gradient(to bottom, #1E222B 50%, #15171E 50%);
        min-height: 100vh;
        overflow-y: auto;
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #15171E;
    }
    ::-webkit-scrollbar-thumb {
        background: #3A3F4B;
        border-radius: 4px;
    }
    body {
        overflow-y: auto;
    }
    .header-section {
        background-color: #15171E;
        padding: 20px 20px;
        text-align: center;
        border-bottom: 1px solid #3A3F4B;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border: none;
        border-radius: 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #3B7BE8;
    }
    h1 {
        color: #FFFFFF;
        margin: 0;
        padding-bottom: 10px;
    }
    h2 {
        color: #FFFFFF;
    }
    h3 {
        color: #FFFFFF;
    }
    .metric-card {
        background-color: #252A33;
        border: 1px solid #3A3F4B;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    .footer {
        text-align: center;
        color: #888;
        font-size: 12px;
        padding: 20px;
        border-top: 1px solid #3A3F4B;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #15171E;
    }
</style>
""", unsafe_allow_html=True)

# Initialize transform title
transform_title = ""

# Header - always show simple header
st.markdown("""
<div class='header-section'>
    <h1>Image Negative Transform</h1>
    <p style='color: #666; font-size: 14px; margin: 5px 0;'>COM 508 Advanced Digital Image Processing</p>
</div>
""", unsafe_allow_html=True)

# File upload
st.markdown("<div style='padding-top: 30px;'></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")

if uploaded_file is not None:
    try:
        # Read the uploaded file (using same logic as negative.py)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Check if image is grayscale (single channel) - same as negative.py
            if len(image.shape) == 2:
                mode = "grayscale"
                transform_title = "Grayscale Image into Negative Image"
            else:
                mode = "rgb"
                transform_title = "Colored Image into Negative Image"
            
            # Display transformation title in two boxes with arrow
            title_parts = transform_title.split(" into ")
            if len(title_parts) == 2:
                st.markdown(f"""
                <div style='display: flex; gap: 10px; justify-content: center; align-items: center; margin: 20px 0;'>
                    <div style='background-color: #252A33; border: 1px solid #3A3F4B; padding: 15px 30px; color: #FFFFFF; font-size: 18px;'>
                        {title_parts[0]}
                    </div>
                    <div style='color: #FFFFFF; font-size: 24px; font-weight: bold;'>→</div>
                    <div style='background-color: #252A33; border: 1px solid #3A3F4B; padding: 15px 30px; color: #FFFFFF; font-size: 18px;'>
                        {title_parts[1]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: #FFFFFF; text-align: center; margin: 20px 0;'>{transform_title}</h2>", unsafe_allow_html=True)
            
            # Apply negative transformation - same formula as negative.py
            L = 256
            negative = (L - 1) - image
            
            # Display side-by-side images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3 style='color: #FFFFFF; font-size: 18px;'>Original</h3>", unsafe_allow_html=True)
                if len(image.shape) == 2:
                    st.image(image, use_container_width=True)
                else:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.markdown("<h3 style='color: #FFFFFF; font-size: 18px;'>Negative</h3>", unsafe_allow_html=True)
                if len(negative.shape) == 2:
                    st.image(negative, use_container_width=True)
                else:
                    st.image(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Download button
            negative_pil = Image.fromarray(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB) if len(negative.shape) == 3 else negative)
            buf = io.BytesIO()
            negative_pil.save(buf, format="PNG")
            buf.seek(0)
            
            st.download_button(
                label="Download negative",
                data=buf,
                file_name="negative_image.png",
                mime="image/png"
            )
            
        else:
            st.error("Error: Could not read the uploaded image.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Add spacing for fixed footer
st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    © 2026 Sondos Deeb
</div>
""", unsafe_allow_html=True)
