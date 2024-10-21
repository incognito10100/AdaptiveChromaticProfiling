import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import gc

# ---------------------------------------------
# Function to Load and Preprocess Image
# ---------------------------------------------
def load_image(image, max_size=800, sample_size=10000):
    """
    Load and preprocess the uploaded image, including resizing and pixel sampling.
    """
    # Load image as an OpenCV image and convert from BGR to RGB
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image if too large
    height, width = img.shape[:2]
    scaling_factor = max_size / max(height, width)
    if scaling_factor < 1:
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        img = cv2.resize(img, new_size)
    
    # Reshape image to a list of pixels
    img = img.reshape((-1, 3))

    # Randomly sample pixels for memory efficiency
    if len(img) > sample_size:
        indices = np.random.choice(len(img), sample_size, replace=False)
        img = img[indices]
    
    return img

# ---------------------------------------------
# Function to Detect Dominant Colors
# ---------------------------------------------
def detect_colors(image_pixels, num_colors):
    """
    Perform K-means clustering to detect dominant colors.
    """
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(image_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# ---------------------------------------------
# Function to Plot Detected Colors
# ---------------------------------------------
def plot_colors(colors):
    """
    Plot the detected dominant colors as a horizontal bar.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    ax.imshow([colors / 255.0])
    ax.axis('off')
    st.pyplot(fig)

# ---------------------------------------------
# Streamlit App
# ---------------------------------------------
def main():
    st.title('🎨 Dominant Color Detection')

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='📷 Uploaded Image', use_column_width=True)
        
        # Load and preprocess image
        image_bytes = uploaded_file.read()
        image_pixels = load_image(image_bytes, max_size=800, sample_size=10000)

        # Select number of dominant colors
        num_colors = st.slider("Select number of dominant colors", min_value=1, max_value=10, value=5)
        
        # Detect dominant colors
        colors = detect_colors(image_pixels, num_colors)

        # Plot detected colors
        st.write("Detected dominant colors:")
        plot_colors(colors)

        # Clean up memory
        del image_pixels, colors
        gc.collect()  # Trigger garbage collection to free memory

# Run the Streamlit app
if __name__ == "__main__":
    main()
