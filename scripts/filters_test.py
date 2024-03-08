from PIL import Image, ImageDraw, ImageEnhance
import numpy as np

def add_noise(image, noise_factor=20):
    # Convert image to NumPy array for easier manipulation
    img_array = np.array(image)

    # Get image shape
    height, width, channels = img_array.shape

    # Generate random noise and scale it by the specified factor
    noise = np.random.normal(scale=noise_factor, size=(height, width, channels))

    # Add noise to the image
    noisy_image = img_array + noise

    # Clip the pixel values to the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to PIL Image
    noisy_image = Image.fromarray(np.uint8(noisy_image))

    return noisy_image

def darken_image(image, factor=1.5):
    # Create an ImageEnhance object
    enhancer = ImageEnhance.Brightness(image)

    # Darken the image using the specified factor
    darkened_image = enhancer.enhance(factor)

    return darkened_image

# Open the image
image_path = "../Augmentation1/All/1_0_0_20161219193326339.jpg.chip.jpg"
original_image = Image.open(image_path)

# Add noise to the image
#noisy_image = add_noise(original_image, noise_factor=25)
darken = darken_image(original_image)
darken.show()
# Display or save the noisy image
#noisy_image.show()
# or save it to a new file
#noisy_image.save("path/to/save/noisy_image.jpg")
