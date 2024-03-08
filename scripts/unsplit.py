import os, shutil

path = "../80_10_10_augprep/Train"
out_path = "../80_10_10_augprep/Train_augprepmerged"

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(root, filename)
            shutil.move(file_path, out_path)
