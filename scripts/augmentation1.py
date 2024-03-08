import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from itertools import product
import random
import numpy as np
def draw_histogram(data, labelx, labely):
    plt.hist(data, bins=range(min(data), max(data) + 2), align='left', rwidth=0.8)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.xticks(range(min(data), max(data) + 1, 5), rotation='horizontal')
    plt.show()


def horizontal_flip(image_path):
    img = Image.open(image_path)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    filename, extension = os.path.splitext(os.path.basename(image_path))
    # Get the directory of the original image
    directory = os.path.dirname(image_path)

    # Create a new filename with "_aug" appended
    new_filename = os.path.join(directory, f"{filename}_flip{extension}")

    img.save(new_filename)

def darken(image_path, factor=0.5):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    darkened_image = enhancer.enhance(factor)
    filename, extension = os.path.splitext(os.path.basename(image_path))
    # Get the directory of the original image
    directory = os.path.dirname(image_path)

    # Create a new filename with "_aug" appended
    new_filename = os.path.join(directory, f"{filename}_dark{extension}")
    darkened_image.save(new_filename)

def brighten(image_path, factor=1.5):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    bright_image = enhancer.enhance(factor)
    filename, extension = os.path.splitext(os.path.basename(image_path))
    # Get the directory of the original image
    directory = os.path.dirname(image_path)

    # Create a new filename with "_aug" appended
    new_filename = os.path.join(directory, f"{filename}_bright{extension}")
    bright_image.save(new_filename)


def blur(image_path):
    img = Image.open(image_path)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    filename, extension = os.path.splitext(os.path.basename(image_path))
    # Get the directory of the original image
    directory = os.path.dirname(image_path)

    # Create a new filename with "_aug" appended
    new_filename = os.path.join(directory, f"{filename}_blur{extension}")
    img.save(new_filename)

def add_noise(image_path, noise_factor=20):
    # Convert image to NumPy array for easier manipulation
    image = Image.open(image_path)
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

    filename, extension = os.path.splitext(os.path.basename(image_path))
    # Get the directory of the original image
    directory = os.path.dirname(image_path)

    # Create a new filename with "_aug" appended
    new_filename = os.path.join(directory, f"{filename}_noise{extension}")
    noisy_image.save(new_filename)

def mixup_augmentation(image_path1, image_path2, alpha=0.2):

    # Load images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    image1 = np.array(image1)
    image2 = np.array(image2)

    # if alpha > 0:
    #     lam = np.random.beta(alpha, alpha)
    # else:
    #     lam = 1
    lam = 0.5
    mixed_image = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
    mixed_img = Image.fromarray(mixed_image)
    filename, extension = os.path.splitext(os.path.basename(image_path1))
    # Get the directory of the original image
    directory = os.path.dirname(image_path1)

    # Create a new filename with "_aug" appended
    new_filename = os.path.join(directory, f"{filename}_mixed{extension}")

    mixed_img.save(new_filename)

def augment(counts, path):
    augmentations = ['flip', 'noise', 'blur', 'darker', 'brighten', 'mixup']
    files = os.listdir(path)
    for age in range (1,91):
        print(f"Processing {age}")
        examples_to_add = max(0, min(counts[age] * 3, 400) - counts[age])
        age_str = f'{age}_'
        curr_age_files = []
        if age == 90:
            senior_ages = ("90_", "91_", "92_", "93_", "94_", "95_", "96_", "97_", "98_", "99_",
                           "100_", "101_", "102_", "103_", "104_", "105_", "106_", "107_", "108_",
                           "109_", "110_", "111_", "112_", "113_", "114_", "115_", "116_", "117_", "118_", "119_")
            curr_age_files = [string for string in files if string.startswith(senior_ages)]
        else:
            curr_age_files = [string for string in files if string.startswith(age_str)]
        pairs = list(product(curr_age_files, augmentations))

        to_add = random.sample(pairs, examples_to_add)

        for filename, augmentation in to_add:
            file_path = f"{path}/{filename}"
            if augmentation == 'flip':
                horizontal_flip(file_path)
            elif augmentation == 'noise':
                add_noise(file_path)
            elif augmentation == 'blur':
                blur(file_path)
            elif augmentation == 'darker':
                darken(file_path)
            elif augmentation == 'brighten':
                brighten(file_path)
            elif augmentation == 'mixup':
                samples = random.sample(curr_age_files, 2)
                if samples[0] == filename:
                    second_file = samples[1]
                else:
                    second_file = samples[0]
                second_path = f"{path}/{second_file}"
                mixup_augmentation(file_path, second_path)


def main():
    folder_path = '../80_10_10_augprep/Train_augprep'  # path to your UTKFace folder

    # Iterate through files in the folder
    age = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            tmp = filename.split('_')
            age.append(min(90, int(tmp[0])))

    counts = [0] * 91
    for person in age:
        counts[person] += 1

    for i in range(1, 91):
        print(f"{i}: {counts[i]}")

    augment(counts, folder_path)
    # Draw histogram
    draw_histogram(age, 'age', 'counts')

if __name__ == "__main__":
    main()
