import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

st.title("📸 Image Processing & Analysis Toolkit")
st.markdown("Streamlit + OpenCV GUI | **Upload → Process → Save**")

# Global image holder
original_image = None
processed_image = None

# Sidebar - Upload
st.sidebar.header("📂 File Menu")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])

# Helper: Load image
def load_image(image_file):
    image = Image.open(image_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Display image info
def image_info(image):
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size_kb = len(cv2.imencode('.jpg', image)[1]) / 1024
    return {
        "Resolution": f"{w} x {h}",
        "Channels": channels,
        "Size (KB)": f"{size_kb:.2f}",
        "Format": uploaded_file.type if uploaded_file else "N/A"
    }

# Color conversions
def convert_color(image, conversion):
    if conversion == "RGB to GRAY":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif conversion == "RGB to HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif conversion == "RGB to YCbCr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif conversion == "BGR to RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Transformation functions
def rotate_image(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def resize_image(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)

def translate_image(image, tx, ty):
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

# Main Panel
if uploaded_file:
    original_image = load_image(uploaded_file)
    processed_image = original_image.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("Processed Image")
        # Sidebar - Choose operation
        st.sidebar.header("🛠️ Operations")

        operation = st.sidebar.selectbox(
            "Select Operation",
            ["None", "Color Conversion", "Rotate", "Resize", "Translate"]
        )

        if operation == "Color Conversion":
            conv_type = st.sidebar.selectbox("Convert", ["RGB to GRAY", "RGB to HSV", "RGB to YCbCr", "BGR to RGB"])
            processed_image = convert_color(original_image, conv_type)

        elif operation == "Rotate":
            angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
            processed_image = rotate_image(original_image, angle)

        elif operation == "Resize":
            scale = st.sidebar.slider("Scale (%)", 10, 300, 100) / 100.0
            processed_image = resize_image(original_image, scale)

        elif operation == "Translate":
            tx = st.sidebar.slider("Shift X", -200, 200, 0)
            ty = st.sidebar.slider("Shift Y", -200, 200, 0)
            processed_image = translate_image(original_image, tx, ty)

        # Display processed image
        if len(processed_image.shape) == 2:
            st.image(processed_image, use_column_width=True, clamp=True)
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Status bar
        st.markdown("---")
        st.subheader("📊 Image Info")
        info = image_info(original_image)
        st.write(info)

        # Save button
        st.download_button(
            label="💾 Download Processed Image",
            data=cv2.imencode('.png', processed_image)[1].tobytes(),
            file_name="processed_image.png",
            mime="image/png"
        )

else:
    st.warning("📤 Please upload an image to get started!")

