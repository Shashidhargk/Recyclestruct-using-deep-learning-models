import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import json

# Load the trained classification model
classifier = tf.keras.models.load_model("/content/drive/MyDrive/material_classifier.h5")

# Class labels
class_names = [
    "Burnt Clay Brick - First Grade", "Burnt Clay Brick - Second Grade", "Burnt Clay Brick - Third Grade",
    "Solid Concrete Block - First Grade", "Solid Concrete Block - Second Grade", "Solid Concrete Block - Third Grade"
]

# Groq API Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to classify the uploaded image
def classify_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch processing

    predictions = classifier.predict(img_array)
    predicted_label = class_names[np.argmax(predictions)]  # Get the highest probability class
    
    return predicted_label

# Function to generate dynamic recommendations using Groq API
def generate_recommendation(material_class):
    prompt = f"""
    You are an expert in sustainable construction and material reuse. Based on the detected material: **{material_class}**, provide a thoughtful and practical recommendation covering:

    1️⃣ **Best Use Cases**: Suggest realistic applications where this material can be effectively reused.
    2️⃣ **Sustainability Impact**: Explain how reusing this material contributes to eco-friendly construction.
    3️⃣ **Enhancement Tips**: Provide simple improvements (e.g., treatments, coatings) to increase durability or usability.
    4️⃣ **Creative Repurposing Ideas**: Offer unique ways this material can be repurposed beyond traditional construction.
    
    {"If the material is classified as 'Third Grade' (Burnt Clay Brick - Third Grade or Solid Concrete Block - Third Grade), provide additional tips on recycling methods, such as crushing for road base material, mixing into new bricks, or using as aggregate in concrete production."}
    
    **Make the response engaging, professional, and tailored to the material type.**
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",  # Groq model
        "messages": [
            {"role": "system", "content": "You are an expert in construction material reuse and sustainability."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠ API Error: {response.text}"

# Streamlit UI
st.title("Material Classification & Smart Recommendations")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Classify the image
    classification = classify_image(image)

    # Generate recommendation using Groq API
    recommendation = generate_recommendation(classification)

    # Display classification result and AI-generated recommendation
    st.write("### Classification Result:")
    st.write(f"🔹 **Predicted Material:** {classification}")
    st.write("### AI-Generated Recommendation:")
    st.write(f"💡 {recommendation}")
