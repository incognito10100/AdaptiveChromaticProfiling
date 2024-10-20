import cv2
import numpy as np
from sklearn.cluster import KMeans
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import os

app = FastAPI()

# Function to load and preprocess the image
def load_image(image):
    img = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
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

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_pixels = load_image(image_bytes)
    colors = detect_colors(image_pixels, 5)

    # Create output image
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)[..., ::-1])
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    for j in range(5):
        plot_colors(colors[j:j+1], axes[j + 1])
        axes[j + 1].set_title(f'Color {j + 1}')

    plt.tight_layout()
    output_path = 'dominant_colors_output.jpg'
    plt.savefig(output_path)
    plt.close()

    return FileResponse(output_path, media_type='image/jpeg')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
