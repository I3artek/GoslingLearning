from PIL import Image, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_sharpness(img):
    # Convert the image to grayscale
    img_gray = img.convert("L")

    # Apply the CONTOUR filter to emphasize edges
    # Calculate the standard deviation of pixel intensities in the contour image
    img_gray = np.array(img_gray)
    img_gray = img_gray / 255

    img_gray = img_gray * img_gray

    img_gray = img_gray * 255
    sharpness_mean = np.mean(img_gray)
    sharpness_std = np.std(img_gray)
    return (sharpness_mean, sharpness_std)

def accumulate_sharpness(folder_path):
    # Initialize empty list for sharpness values
    sharpness_values_mean = []
    sharpness_values_std = []


    # Get a list of image files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
                  and 70 <= int(f.split('_')[0]) <= 100]

    # Iterate through each image and accumulate sharpness values
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Open the image
        img = Image.open(image_path)

        # Calculate and accumulate sharpness
        sharpness_mean, sharpness_std = calculate_sharpness(img)
        sharpness_values_mean.append(sharpness_mean)
        sharpness_values_std.append(sharpness_std)

    return (sharpness_values_mean, sharpness_values_std)

def plot_sharpness_histogram(sharpness_values, type):
    plt.hist(sharpness_values, bins=256, color='orange', alpha=0.7, density=True, range=(0, 255))
    plt.title(f'{type} of contour pixel intensity histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(0,255)
    plt.ylim(0, 0.1)
    plt.show()

def main(folder_path):
    # Generate and accumulate sharpness values
    sharpness_values_mean, sharpness_values_std = accumulate_sharpness(folder_path)

    # Plot sharpness histogram
    plot_sharpness_histogram(sharpness_values_mean, "Mean")
    plot_sharpness_histogram(sharpness_values_std, "Std")

if __name__ == "__main__":
    folder_path = r'../Cleanup3/UTKFaceCleanup3'
    main(folder_path)
