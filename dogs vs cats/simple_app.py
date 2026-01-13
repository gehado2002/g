import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐶🐱",
    layout="centered"
)

st.title("🐶🐱 Dogs vs Cats Classification")
st.write("Upload an image and let the AI predict whether it's a Dog or a Cat.")

# ----------------------------
# Load Model
# ----------------------------
# Make sure your model file (best_model.keras) is in the same repo folder
try:
    model = load_model("best_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ----------------------------
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # ----------------------------
    # Preprocess Image
    # ----------------------------
    img_resized = img.resize((150, 150))  # same size used in your CNN
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # Make Prediction
    # ----------------------------
    try:
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]

        if class_index == 0:
            st.success("This is a **Cat** 🐱")
        else:
            st.success("This is a **Dog** 🐶")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
