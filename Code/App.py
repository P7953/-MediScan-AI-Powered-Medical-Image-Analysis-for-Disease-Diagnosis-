import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the pre-trained model (replace with your model if needed)
model = load_model('trained_model.h5')  # Adjust the model path as needed

# Class labels
class_labels = ['Normal', 'Glaucoma', 'Cataract', 'Normal']  # Replace with your actual class labels


# Function to preprocess and predict the image
def predict_image(uploaded_file):
    # Open image file
    uploaded_image = Image.open(uploaded_file)

    # Resize image to (299, 299) as required by InceptionV3
    uploaded_image = uploaded_image.resize((299, 299))

    # Convert image to array
    image_array = image.img_to_array(uploaded_image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocess the image for InceptionV3
    image_array = preprocess_input(image_array)

    # Predict using the model
    predictions = model.predict(image_array)

    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions)  # Index of the max probability
    confidence = np.max(predictions)  # Maximum confidence

    return predicted_class, confidence


# Streamlit UI
st.title("MediScan - AI Powered Disease Diagnosis")
st.write("Upload an image for prediction")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When the user uploads an image
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Show a message while the image is being analyzed
    st.write("Analyzing the image...")

    # Call the prediction function
    predicted_class, confidence = predict_image(uploaded_file)

    # Show prediction result
    st.write(f"Prediction: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
