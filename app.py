import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_image(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img.reshape((-1, 3))  # Reshape to a list of pixels
    return img

# Function to perform K-means clustering to detect dominant colors
def detect_colors(image_pixels, num_colors):
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Function to plot the colors
def plot_colors(colors):
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    ax.imshow([colors / 255.0])
    ax.axis('off')
    st.pyplot(fig)

# Streamlit app
st.title('Dominant Color Detection')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    image_bytes = uploaded_file.read()
    image_pixels = load_image(image_bytes)
    
    # Detect dominant colors
    num_colors = st.slider("Select number of dominant colors", min_value=1, max_value=10, value=5)
    colors = detect_colors(image_pixels, num_colors)

    # Plot and display dominant colors
    st.write("Detected dominant colors:")
    plot_colors(colors)
