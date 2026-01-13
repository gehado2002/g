import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# ----------------------------
# تحميل الموديل من Google Drive لو مش موجود محلياً
# ----------------------------
model_path = "vgg16_best_model.keras"
gdrive_url = "https://drive.google.com/uc?id=1X-OXVhF_2sIv2FDGXVIQ_oRSo4HnFc9H"

# تحميل الموديل مرة واحدة عند بدء التطبيق
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(model_path):
        st.info("تحميل الموديل من Google Drive ...")
        gdown.download(gdrive_url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon=":dog: :cat:",  # بدل الإيموجي المباشر
    layout="centered"
)

st.title("Dogs vs Cats Classification")
st.write("Upload an image and let the AI decide whether it's a Dog or a Cat.")

# رفع صورة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(image.resize((150,150)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"
    st.success(f"Prediction: {result}")
