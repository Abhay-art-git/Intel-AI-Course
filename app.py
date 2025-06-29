import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model('model/trained_model.h5')

st.title("Visual Quality Check System")

option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

def predict(img):
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Defective" if prediction > 0.5 else "Good"

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Packaging Image", type=['jpg', 'png'])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        result = predict(img)
        st.success(f"Result: {result}")

elif option == "Use Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if st.button("Start Webcam"):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            stframe.image(frame, channels="BGR")
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
