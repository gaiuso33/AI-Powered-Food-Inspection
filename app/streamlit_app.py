import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('fresh_stale_classifier.h5')

st.title('Fresh or Stale Classifier 🍎🍌')

# Upload an image
uploaded_file = st.file_uploader("Choose a fruit or vegetable image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.write('Prediction: **Stale** ❌')
    else:
        st.write('Prediction: **Fresh** ✅')
