import shutil
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import mimetypes
import os
import numpy as np
from modeluj import preprocess
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# some global variables

# some state I guess?
paused = False
# count of frames for something?
saved_frames_count = 0
# in the next lines the coordinates denote the central point of each object
# average coordinates of left eye in utk faces
left_eye_utk = (57, 59)
# average coordinates of right eye in utk faces
right_eye_utk = (135, 59)
# average coordinates of mounth in utk faces
mouth_utk = (97, 155)
# average distance between the eyes
distance_utk = right_eye_utk[0] - left_eye_utk[0]
# midpoint of the eyes
utk_midpoint = ((left_eye_utk[0] + right_eye_utk[0]) // 2,
                (left_eye_utk[1] + right_eye_utk[1]) // 2)
# some magic with maps (?)
atoi_map = [str(x) for x in range(1, 91)]
atoi_map.sort()
atoi_map = [int(x) for x in atoi_map]

# hardcoded biases
biases = [0,0,1,0,-1,0,-1,-2,-2,0,1,0,2,-1,3,2,2,3,1,1,2,
          3,0,0,1,0,-2,2,-1,0,0,0,1,1,4,3,-1,1,0,-2,1,2,
          2,3,2,3,0,-3,0,5,4,3,2,1,1,0,2,3,8,8,3,5,6,2,
          1,4,1,-1,0,-1,-1,0,0,4,3,0,1,9,2,0,-6,0,-2,1,1,-1,-2,0,0,0]

def load_model(model_path):
    """Load the desired model."""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)

    model.fc = torch.nn.Sequential(
       torch.nn.Linear(model.fc.in_features, 512),
       torch.nn.ReLU(),
       torch.nn.Dropout(0.5),
       torch.nn.Linear(512, 90)
    )
    # Load the pre-trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  # only difference


def preprocess_image(image):
    """Preprocess an image so that it's suitable for our model."""
    # transform = transforms.ToTensor()
    transform = transforms.Compose([
         transforms.Resize((206, 206)),
         transforms.Grayscale(),
         transforms.ToTensor(),
     ])
    image = transform(image).unsqueeze(0)
    return image


def predict(model, image_tensor):
    """Predict age from image using loaded model."""
    with torch.no_grad():
        outputs = model(image_tensor).numpy()
        outputs = softmax(outputs)
        entry_pred = np.sum(np.multiply(outputs, [int(x) for x in atoi_map])).astype(np.int8)
        return entry_pred - biases[entry_pred - 1]


# let's load the model so we can have it as a global variable
model = load_model('model.pth')


# wrapper function for the whole process image -> number
# image is a cropped face
def calculate_age(image):
    """Return age from image."""
    return predict(model, preprocess_image(preprocess(image)))


def remove_bounded_images_folder():
    """Remove folder with images containing bounded boxes."""
    bounded_images_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'boundedImages')
    if os.path.exists(bounded_images_folder):
        shutil.rmtree(bounded_images_folder)


# loading the classifiers with respected files
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("./haarcascade_mcs_mouth.xml")


def get_faces_from_image(gray):
    """Detect faces in a grayscale image."""
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=15)
    return faces


def align_with_point(image, utkPoint, imagePoint):
    """Align an image so that imagePoint has the same coords as utkPoint."""
    dX = imagePoint[0] - utkPoint[0]
    dY = imagePoint[1] - utkPoint[1]

    new_image = np.zeros((200, 200, 3), dtype=np.uint8)
    n, m = image.shape[:2]
    N, M = new_image.shape[:2]
    # we need 0<=i<N, 0<=j<M, 0<=i+dY<n, 0<=j+dX<m
    # that's equivalent to:
    # max(0,-dY)<=i<min(N,n-dY), max(0,-dX)<=j<min(M,m-dX)
    # so:
    new_image[max(0, -dY):min(N, n - dY),
              max(0, -dX):min(M, m - dX)] = image[max(dY, 0):min(N + dY, n),
                                                  max(dX, 0):min(M + dX, m)]
    return new_image


def check_scale(image_width, distance):
    """Check if the scale is good."""
    if distance / image_width < distance_utk / 200 - 0.1:
        return False
    return True


def align_bartek(x, y, w, h, gray_image, original_image, outputSize=(200, 200)):
    """Align the photo so that it matches utk face."""
    face_unchanged = original_image[y:y + h, x:x + w]

    # Divide the image
    gray_top = gray_image[y:y + h // 2, x:x + w]

    # detect the eyes
    eyes = eye_cascade.detectMultiScale(gray_top, scaleFactor=1.02,
                                        minNeighbors=10, minSize=(25, 25))

    if len(eyes) == 2:
        # if there are exactly two eyes on the image
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[:2]
        # Calculate the center of each eye
        eye_centers = [(x + x1 + w1 // 2, y + y1 + h1 // 2),
                       (x + x2 + w2 // 2, y + y2 + h2 // 2)]

        # Sort the eye centers based on x-coordinate
        eye_centers.sort()

        left_eye_center = eye_centers[0]
        right_eye_center = eye_centers[1]
        distance = np.sqrt(
            (right_eye_center[0] - left_eye_center[0]) ** 2 + (right_eye_center[1] - left_eye_center[1]) ** 2)

        if check_scale(w, distance):
            # Rotate so eye centers are aligned
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]

            angle = np.degrees(np.arctan2(dY, dX))
            eyes_center = ((float)(left_eye_center[0] + right_eye_center[0]) // 2,
                           (float)(left_eye_center[1] + right_eye_center[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            rotated_image = cv2.warpAffine(original_image, M,
                                           (original_image.shape[1], original_image.shape[0]),
                                           flags=cv2.INTER_CUBIC)
            # cv2.imshow('Rotated', rotated_image)
            # cv2.imwrite(f'boundedImages/rotated_image_{saved_frames_count}.png',
            #            rotated_image)

            # Scale so distance between eyes is the same as in utk
            scale = distance_utk / distance
            scaled_image = cv2.resize(rotated_image, None, fx=scale,
                                      fy=scale, interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('Scaled', scaled_image)

            # cv2.imwrite(f'boundedImages/scaled_image_{saved_frames_count}.png',
            #            scaled_image)

            left_eye_center = (int(left_eye_center[0] * scale),
                               int(left_eye_center[1] * scale))
            # Move so left eye is in the same place as in utk
            transformed_image = align_with_point(scaled_image,
                                                 left_eye_utk, left_eye_center)
            # cv2.imwrite(
            #    f'boundedImages/transformed_image_{saved_frames_count}.png',
            #    transformed_image)
            # cv2.imshow('Transformed', transformed_image)
            return transformed_image

    # if we don't have exactly two eyes we check the bottom third of the image
    gray_bottom = gray_image[y + 2 * h // 3:y + h, x:x + w]
    # detect the mouth
    mouths = mouth_cascade.detectMultiScale(gray_bottom, scaleFactor=1.1,
                                            minNeighbors=8, minSize=(30, 50))
    if len(mouths) == 1:
        # if we have one mouth then we can try to align with it
        mouth_center = (x + mouths[0][0] + mouths[0][2] // 2,
                        y + mouths[0][1] + mouths[0][3] // 2 + 2 * h // 3)
        transformed_image = align_with_point(original_image,
                                             mouth_utk, mouth_center)
        return transformed_image
    # any other case
    return cv2.resize(face_unchanged, outputSize,
                      interpolation=cv2.INTER_AREA)


def process_frame(image):
    """Process one frame."""
    # global saved_frames_count
    # bounded_images_folder = os.path.join(
    #    os.path.dirname(os.path.abspath(__file__)), 'boundedImages')
    # if not os.path.exists(bounded_images_folder):
    #    os.makedirs(bounded_images_folder)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = get_faces_from_image(gray)

    face_rectangles = []
    face_ages = []

    for i, (x, y, w, h) in enumerate(faces):
        aligned_face_resized = align_bartek(x, y, w, h, gray, image)
        aligned_pil = Image.fromarray(aligned_face_resized)
        age = calculate_age(aligned_pil)

        face_rectangles.append((x, y, x + w, y + h))
        face_ages.append(age)

    for (x1, y1, x2, y2), age in zip(face_rectangles, face_ages):
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if (y1 > 30):
            cv2.putText(image, f"{age}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            cv2.putText(image, f"{age}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # cv2.imwrite(
    #    f'{bounded_images_folder}/video_frame_{saved_frames_count}.png', image)
    # saved_frames_count += 1
    return image


def video_feed():
    """Process live video feed."""
    # Create a window with the resizable flag
    cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)

    # Set the initial size of the window
    cv2.resizeWindow('Video Feed', 640, 480)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow('Video Feed', frame)
        # Set the desired framerate
        framerate = 60
        delay = int(1000 / framerate)  # in milliseconds
        key = cv2.waitKey(delay)
        if key == ord('q') or cv2.getWindowProperty('Video Feed', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image_internal(file_path):
    """Return an image from a path."""
    image = cv2.imread(file_path)

    aspect_ratio = image.shape[1] / image.shape[0]

    max_display_width = 800
    max_display_height = 600

    if aspect_ratio > 1:
        display_width = min(image.shape[1], max_display_width)
        display_height = int(display_width / aspect_ratio)
    else:
        display_height = min(image.shape[0], max_display_height)
        display_width = int(display_height * aspect_ratio)

    image = cv2.resize(image, (display_width, display_height))
    image = process_frame(image)
    return image


def process_image():
    """Process an image saved on disk."""
    file_path = filedialog.askopenfilename()
    if file_path:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None or not mime_type.startswith('image'):
            messagebox.showerror("Error", "Selected file is not an image")
            return
        image = process_image_internal(file_path)
        cv2.imshow('Detected Faces', image)
        while cv2.waitKey(1000) < 0:
            if cv2.getWindowProperty('Detected Faces',
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def process_images_folder():
    """Process all images in a directory."""
    # get gir
    dir_path = filedialog.askdirectory()
    if dir_path:
        # create directory for results
        results_dir = f"{dir_path}_results"
        os.mkdir(results_dir)
        file_names = os.listdir(dir_path)
        # iterate through files
        for name in file_names:
            file_path = os.path.join(dir_path, name)
            # check type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None or not mime_type.startswith('image'):
                continue
            image = process_image_internal(file_path)
            # save the image
            cv2.imwrite(os.path.join(results_dir, name), image)
        pass


def process_video():
    """Process a video saved on disk."""
    global paused, cap
    file_path = filedialog.askopenfilename()

    if file_path:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None or not mime_type.startswith('video'):
            messagebox.showerror("Error", "Selected file is not a video")
            return
        cap = cv2.VideoCapture(file_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Define the output file and create a VideoWriter object with the same codec
        output_file = file_path.replace('.', '_result.')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not paused:
                frame = process_frame(frame)
                cv2.imshow(
                    'Detected Faces (Press "q" to exit, "p" to pause/resume)',
                    frame)
                out.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q') or cv2.getWindowProperty(
                    'Detected Faces (Press "q" to exit, "p" to pause/resume)',
                    cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('p'):
                paused = not paused

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def create_gui():
    """Create the gui."""
    root = tk.Tk()
    root.title("Face Detection App")
    root.geometry("800x600")

    def configure_button(button):
        button.config(font=('Arial', 20), bg='blue', fg='white', padx=20, pady=10)

    video_button = tk.Button(root, text="Start Video Feed", command=video_feed)
    configure_button(video_button)
    video_button.pack(pady=20)

    image_button = tk.Button(root, text="Process Image", command=process_image)
    configure_button(image_button)
    image_button.pack(pady=20)

    video_process_button = tk.Button(root, text="Process Video", command=process_video)
    configure_button(video_process_button)
    video_process_button.pack(pady=20)

    images_folder_process_button = tk.Button(root, text="Process Folder with Images", command=process_images_folder)
    configure_button(images_folder_process_button)
    images_folder_process_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    remove_bounded_images_folder()
    create_gui()

