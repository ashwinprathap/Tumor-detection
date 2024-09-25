import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model("cnn_tumor.h5")

# Function to make predictions
def make_prediction(img, model):
    # Resize the image to 128x128 (as per your model)
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    
    # Convert the image to an array and expand dimensions to match model input
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    
    # Normalize the image (as per your training process)
    img = tf.keras.utils.normalize(img, axis=1)
    
    # Make prediction
    res = model.predict(img)
    
    # Return result based on prediction output
    if res[0][0] > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

# Streamlit UI
st.title("Tumor Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Submit button
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False,width=200)
    
    # Convert the uploaded image to a format compatible with the model
    image_np = np.array(image)
    
    if st.button("Submit"):
        # Make the prediction
        result = make_prediction(image_np, model)
        
        # Display the result
        st.write("Prediction: ", result)
