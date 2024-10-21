import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------------------------------
# Function to Load and Preprocess Image
# ---------------------------------------------
def load_image(image_path, max_size=800):
    """
    Load an image from file, resize if needed, and reshape to a list of pixels.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Resize if image is larger than max_size
    height, width = img.shape[:2]
    scaling_factor = max_size / max(height, width)
    if scaling_factor < 1:
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        img = cv2.resize(img, new_size)
    
    # Reshape to a list of pixels
    img = img.reshape((-1, 3))
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
def plot_colors(colors, ax):
    """
    Plot a horizontal bar of detected colors.
    """
    ax.imshow([colors / 255.0])
    ax.axis('off')

# ---------------------------------------------
# Process and Save Images with Detected Colors
# ---------------------------------------------
def process_and_save_images(image_paths, num_colors=5):
    """
    Process multiple images, detect dominant colors, and save the results.
    """
    fig, axes = plt.subplots(len(image_paths), num_colors + 1, figsize=(15, len(image_paths) * 3))

    for i, image_path in enumerate(image_paths):
        # Load image
        image_pixels = load_image(image_path)
        
        # Detect dominant colors
        colors = detect_colors(image_pixels, num_colors)
        
        # Load and display the original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i + 1}')
        axes[i, 0].axis('off')
        
        # Display dominant colors
        for j in range(num_colors):
            plot_colors(colors[j:j+1], axes[i, j + 1])
            axes[i, j + 1].set_title(f'Color {j + 1}')
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('dominant_colors_output.jpg')
    plt.show()

# ---------------------------------------------
# Main Function
# ---------------------------------------------
def main():
    """
    Main function to process a list of images and detect dominant colors.
    """
    # List of image paths
    image_paths = [
        'images/sample_image1.jpg',
        'images/sample_image2.jpg',
        'images/sample_image3.jpg'
    ]
    
    # Process and save images with dominant colors
    process_and_save_images(image_paths, num_colors=5)

# Run the main function
if __name__ == "__main__":
    main()
