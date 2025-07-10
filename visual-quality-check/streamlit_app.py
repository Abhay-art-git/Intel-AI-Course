import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("model/quality_model.h5")
classes = ["good", "defective"]

st.set_page_config(page_title="Visual Quality Check")
st.title("📦 Visual Quality Check System")

st.markdown("Upload or capture an image of a package to check its quality.")

# Choose input method
input_type = st.radio("Choose Input Method:", ["📁 Upload Image", "📷 Use Webcam"])

image = None

if input_type == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_type == "📷 Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture).convert("RGB")

# Process if image is available
if image is not None:
    st.image(image, caption="Selected Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    label = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    st.subheader("Prediction:")
    if label == "good":
        st.success(f"✅ Good Package ({confidence:.2f}%)")
    else:
        st.error(f"❌ Defective Package ({confidence:.2f}%)")
