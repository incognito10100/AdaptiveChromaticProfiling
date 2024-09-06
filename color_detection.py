import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_image(image_path):
    img = cv2.imread(image_path)
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
def plot_colors(colors, ax):
    ax.imshow([colors / 255.0])  # Normalize the color value
    ax.axis('off')

def process_and_save_images(image_paths, num_colors=5):
    fig, axes = plt.subplots(len(image_paths), num_colors + 1, figsize=(15, len(image_paths) * 3))
    
    for i, image_path in enumerate(image_paths):
        # Load image
        image_pixels = load_image(image_path)
        
        # Detect colors
        colors = detect_colors(image_pixels, num_colors)
        
        # Load and display the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i + 1}')
        axes[i, 0].axis('off')
        
        # Display dominant colors
        for j in range(num_colors):
            plot_colors(colors[j:j+1], axes[i, j + 1])
            axes[i, j + 1].set_title(f'Color {j + 1}')
    
    plt.tight_layout()
    plt.savefig('dominant_colors_output.jpg')
    plt.show()

def main():
    # List of sample images
    image_paths = [
        'images/sample_image1.jpg',
        'images/sample_image2.jpg',
        'images/sample_image3.jpg'
    ]
    
    # Process and save images with dominant colors
    process_and_save_images(image_paths)

if __name__ == "__main__":
    main()
