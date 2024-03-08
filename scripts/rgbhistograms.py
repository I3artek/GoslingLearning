from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def accumulate_histograms(folder_path):
    # Initialize empty histograms
    total_histogram_red = np.zeros(256)
    total_histogram_green = np.zeros(256)
    total_histogram_blue = np.zeros(256)
    total_histogram_lightness = np.zeros(256)

    # Get a list of image files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Iterate through each image and accumulate histograms
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Open the image
        img = Image.open(image_path)

        # Convert the image to RGB mode (if it's not already in RGB)
        img = img.convert("RGB")

        # Split the image into red, green, and blue channels
        r, g, b = img.split()

        # Accumulate histograms
        total_histogram_red += np.array(r.histogram())
        total_histogram_green += np.array(g.histogram())
        total_histogram_blue += np.array(b.histogram())

        # Convert the image to grayscale
        img_gray = img.convert("L")

        # Accumulate histogram for the lightness channel
        total_histogram_lightness += np.array(img_gray.histogram())

    return total_histogram_red, total_histogram_green, total_histogram_blue, total_histogram_lightness

def plot_combined_histograms(total_histogram_red, total_histogram_green, total_histogram_blue, total_histogram_lightness):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.bar(range(256), total_histogram_red, color='red', alpha=0.7)
    plt.title('Red Channel')

    plt.subplot(2, 2, 2)
    plt.bar(range(256), total_histogram_green, color='green', alpha=0.7)
    plt.title('Green Channel')

    plt.subplot(2, 2, 3)
    plt.bar(range(256), total_histogram_blue, color='blue', alpha=0.7)
    plt.title('Blue Channel')

    plt.subplot(2, 2, 4)
    plt.bar(range(256), total_histogram_lightness, color='gray', alpha=0.7)
    plt.title('Lightness Channel')

    plt.tight_layout()
    plt.show()

def main(folder_path):
    # Generate and accumulate histograms
    total_histogram_red, total_histogram_green, total_histogram_blue, total_histogram_lightness = accumulate_histograms(folder_path)

    # Plot combined histograms
    plot_combined_histograms(total_histogram_red, total_histogram_green, total_histogram_blue, total_histogram_lightness)

if __name__ == "__main__":
    folder_path = '../Group Histograms/70-100'
    main(folder_path)
