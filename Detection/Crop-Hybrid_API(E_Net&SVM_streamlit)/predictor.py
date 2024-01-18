import streamlit as st
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
import efficientnet.tfkeras as efn
import numpy as np

loaded_svm_model = joblib.load('svm_model.joblib')
list_of_classes = ['Rice Brown Spot', 'Corn Common Rust', 'Rice Healthy', 'Corn Healthy', 'Potato Late Blight', 'Potato Healthy']

def load_model():
    enet = efn.EfficientNetB0(
        input_shape=(64, 64, 3),
        weights='imagenet',
        include_top=False
    )
    x = enet.output
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Dense(6, activation='softmax')(x)
    e_model_b0 = tf.keras.Model(inputs=enet.input, outputs=y)
    return e_model_b0

model = load_model()
model.load_weights('efficientnet_b0_model.h5')
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Prediction function
def predict(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0
    pred = model.predict(img_array)
    new_data_flat = pred.reshape(1, -1)
    prediction = loaded_svm_model.predict(new_data_flat)[0]
    return list_of_classes[prediction]

# Streamlit app
st.title("Crop Disease Prediction App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    prediction = predict(uploaded_file)
    if prediction in ['Rice Brown Spot', 'Corn Common Rust', 'Potato Late Blight']:
    # st.write()
    # st.title(f"Prediction: {prediction}")
        st.markdown(f"<h1 style='text-align: center; color: red;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: green;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
