from PIL import Image, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_sharpness(img):
    # Convert the image to grayscale
    img_gray = img.convert("L")

    # Apply the CONTOUR filter to emphasize edges
    contour_img = img_gray.filter(ImageFilter.CONTOUR)

    # Calculate the standard deviation of pixel intensities in the contour image
    sharpness = np.mean(np.array(contour_img))

    return sharpness

def accumulate_sharpness(folder_path):
    # Initialize empty list for sharpness values
    sharpness_values = []

    # Get a list of image files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
                  and 1 <= int(f.split('_')[0]) <= 5]

    # Iterate through each image and accumulate sharpness values
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Open the image
        img = Image.open(image_path)

        # Calculate and accumulate sharpness
        sharpness = calculate_sharpness(img)
        sharpness_values.append(sharpness)

    return sharpness_values

def plot_sharpness_histogram(sharpness_values):
    plt.hist(sharpness_values, bins=30, color='orange', alpha=0.7, density=True)
    plt.title('Mean of contour pixel intensity histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def main(folder_path):
    # Generate and accumulate sharpness values
    sharpness_values = accumulate_sharpness(folder_path)

    # Plot sharpness histogram
    plot_sharpness_histogram(sharpness_values)

if __name__ == "__main__":
    folder_path = r'../Cleanup3/UTKFaceCleanup3'
    main(folder_path)
