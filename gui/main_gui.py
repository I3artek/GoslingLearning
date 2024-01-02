import shutil

import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import mimetypes
import os

paused = False
saved_frames_count = 0

# wrapper function for the whole process image -> number
# image is a cropped face
def calculate_age(image):
    # mockup version
    return 7


def remove_bounded_images_folder():
    bounded_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boundedImages')
    if os.path.exists(bounded_images_folder):
        shutil.rmtree(bounded_images_folder)


def detect_faces(image):
    global saved_frames_count
    bounded_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boundedImages')
    if not os.path.exists(bounded_images_folder):
        os.makedirs(bounded_images_folder)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(60, 60))

    for i, (x, y, w, h) in enumerate(faces):
        bounded_image = image[y:y+h, x:x+w]

        # to trzeba w odpowiednie miejsce przeniesc, bo ja nwm gdzie sie to croppuje
        age = calculate_age(bounded_image)

        cv2.imwrite(f'{bounded_images_folder}/video_frame_{saved_frames_count}_face_{i}.png', bounded_image)

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

        frame = detect_faces(frame)
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
    image = detect_faces(image)
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
        cv2.waitKey(0)
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
                frame = detect_faces(frame)
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
