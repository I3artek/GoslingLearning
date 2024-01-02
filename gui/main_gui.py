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
import dlib



paused = False
saved_frames_count = 0
left_eye_utk = (57, 59)
right_eye_utk = (135, 59)
distance_utk = right_eye_utk[0] - left_eye_utk[0]
utk_midpoint = ((left_eye_utk[0] + right_eye_utk[0]) // 2, (left_eye_utk[1] + right_eye_utk[1]) // 2)


# wrapper function for the whole process image -> number
# image is a cropped face

def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)

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
    #transform = transforms.ToTensor()
    transform = transforms.Compose([
         transforms.Resize((206, 206)),
         transforms.Grayscale(),
         transforms.ToTensor(),
     ])
    image = transform(image).unsqueeze(0)
    return image

# Function to predict using the loaded model
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor).numpy()
        outputs = softmax(outputs)
        return np.sum(np.multiply(outputs, np.arange(1, 91))).astype(np.int8)

model = load_model('resnet18_prep1_checkpoint_epoch_lowest.pth')


# wrapper function for the whole process image -> number
# image is a cropped face
def calculate_age(image):
    return predict(model, preprocess_image(preprocess(image)))

def remove_bounded_images_folder():
    bounded_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boundedImages')
    if os.path.exists(bounded_images_folder):
        shutil.rmtree(bounded_images_folder)

# loading the classifiers with respected files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def get_faces_from_image(gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
    return faces

def align_and_resize_face(image, output_size=(200, 200)):
    # Convert image to grayscale for eye detection
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    if len(eyes) == 2:  # Assuming at least two eyes are detected for alignment
        # Extract the coordinates of the eyes
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[:2]
        # Calculate the center of each eye
        eye1_center = (x1 + w1 // 2, y1 + h1 // 2)
        eye2_center = (x2 + w2 // 2, y2 + h2 // 2)

        # check left and right eye
        left_eye_center = eye1_center if eye1_center[0] < eye2_center[0] else eye2_center
        right_eye_center = eye1_center if eye1_center[0] > eye2_center[0] else eye2_center

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyes_center = ((float)(left_eye_center[0] + right_eye_center[0]) // 2, (float)(left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        # move to left eye

        transformed_image = align_with_point(rotated_image, left_eye_center, left_eye_utk)
        return transformed_image
    else:
        # If less than two eyes are detected, return the original image
        return image


def align_with_point_and_resize(image, imagePoint, utkPoint):
    dX = utkPoint[0] - imagePoint[0]
    dY = utkPoint[1] - imagePoint[1]
    new_img = np.zeros((200, 200, image.shape[2]), dtype=np.uint8)
    for x in range(200):
        for y in range(200):
            src_x = x - dX
            src_y = y - dY
            # Ensure the source coordinates are within the image boundaries
            if 0 <= src_x < image.shape[0] and 0 <= src_y < image.shape[1]:
                new_img[x, y] = image[src_x, src_y]
    return new_img

def align_with_point(image, imagePoint, utkPoint):
    dX = utkPoint[0] - imagePoint[0]
    dY = utkPoint[1] - imagePoint[1]
    new_img = np.zeros((200, 200, image.shape[2]), dtype=np.uint8)
    for x in range(200):
        for y in range(200):
            src_x = x - dX
            src_y = y - dY
            # Ensure the source coordinates are within the image boundaries
            if 0 <= src_x < image.shape[0] and 0 <= src_y < image.shape[1]:
                new_img[x, y] = image[src_x, src_y]
    return new_img


def align_bartek(x, y, w, h, image, outputSize = (200, 200)) :
    face_unchanged = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(face_unchanged, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))
    if len(eyes) == 2:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[:2]
        # Calculate the center of each eye
        eye1_center = (x + x1 + w1 // 2, y + y1 + h1 // 2)
        eye2_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)

        # check left and right eye
        left_eye_center = eye1_center if eye1_center[0] < eye2_center[0] else eye2_center
        right_eye_center = eye1_center if eye1_center[0] > eye2_center[0] else eye2_center

        # rotate
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyes_center = ((float)(left_eye_center[0] + right_eye_center[0]) // 2, (float)(left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
        #cv2.imshow('Rotated Image', rotated_image)
        
        # scale
        distance = np.sqrt((right_eye_center[0] - left_eye_center[0]) ** 2 + (right_eye_center[1] - left_eye_center[1]) ** 2)
        scale = distance_utk / distance
        scaled_image = cv2.resize(rotated_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('Scaled Image', scaled_image)

        # move to left eye
        transformed_image = align_with_point_and_resize(scaled_image, left_eye_center, left_eye_utk)
        cv2.imshow('Transformed Image', transformed_image)
        #result_image = transformed_image[0:200, 0:200]
        #cv2.imshow('result', result_image)
        return transformed_image
    else:
        return cv2.resize(face_unchanged, outputSize, interpolation=cv2.INTER_AREA)



def process_frame(image):
    global saved_frames_count
    bounded_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boundedImages')
    if not os.path.exists(bounded_images_folder):
        os.makedirs(bounded_images_folder)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = get_faces_from_image(gray)

    for i, (x, y, w, h) in enumerate(faces):

        # for (ex, ey, ew, eh) in eyes:
        #    # Draw rectangles around the detected eyes
        #    cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        # align face and calculate age
        aligned_face_resized = align_and_resize_face(image[y:y + h, x:x + w])
        #aligned_face_resized = align_bartek(x, y, w, h, image)
        aligned_pil = Image.fromarray(aligned_face_resized)
        age = calculate_age(aligned_pil)

        # save aligned face image
        cv2.imwrite(f'{bounded_images_folder}/video_frame_{saved_frames_count}_face_{i}.png', aligned_face_resized)

        # Draw rectangle around detected face and put age as text
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    saved_frames_count += 1
    return image


def video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow('Video Feed', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty('Video Feed', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


# return an image from a path
def process_image_internal(file_path):
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


# process an image
def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None or not mime_type.startswith('image'):
            messagebox.showerror("Error", "Selected file is not an image")
            return
        image = process_image_internal(file_path)
        cv2.imshow('Detected Faces', image)
        while cv2.waitKey(1000) < 0:
            if cv2.getWindowProperty('Detected Faces', cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


# process image folder
def process_images_folder():
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
    global paused, cap
    file_path = filedialog.askopenfilename()

    if file_path:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None or not mime_type.startswith('video'):
            messagebox.showerror("Error", "Selected file is not a video")
            return
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not paused:
                frame = process_frame(frame)
                cv2.imshow('Detected Faces (Press "q" to exit, "p" to pause/resume)', frame)

            key = cv2.waitKey(1)
            if key == ord('q') or cv2.getWindowProperty('Detected Faces (Press "q" to exit, "p" to pause/resume)',
                                                        cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('p'):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()

def create_gui():
    root = tk.Tk()
    root.title("Face Detection App")
    root.geometry("800x600")

    def configure_button(button):
        button.config(font=('Comic Sans MS', 20), bg='blue', fg='white', padx=20, pady=10)

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

