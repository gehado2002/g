import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title=Dogs vs Cats Classifier,
    page_icon=🐶🐱,
    layout=centered
)

st.title(🐶🐱 Dogs vs Cats Classification)
st.write(Upload an image and let the AI decide whether it's a Dog or a Cat.)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model()
    model = tf.keras.models.load_model(vgg16_best_model.keras)
    return model

model = load_model()

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image)
    image = image.resize((150, 150))
    img_array = np.array(image)  255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    📤 Upload an image,
    type=[jpg, jpeg, png]
)

if uploaded_file is not None
    image = Image.open(uploaded_file).convert(RGB)
    st.image(image, caption=Uploaded Image, use_column_width=True)

    if st.button(🔍 Predict)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        if prediction  0.5
            st.success(f🐶 Dog ({prediction.2%} confidence))
        else
            st.success(f🐱 Cat ({1 - prediction.2%} confidence))

# ----------------------------
# Footer
# ----------------------------
st.markdown(---)
st.markdown(📌 Model VGG16 Transfer Learning)
st.markdown(👩‍💻 Built with ❤️ using Streamlit)
