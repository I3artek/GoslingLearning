import os, shutil
imagesPath = '../UTKFace'
path = '../ageSplit'

for image in os.listdir(imagesPath):
    age = int(image.split('_')[0])

    if 0 <= age <= 20:
        try:
            shutil.copy(f"{imagesPath}/{image}", f"{path}/020")
        except Exception as e:
            print(f"An error occurred: {e}")
    elif 21 <= age <= 40:
        try:
            shutil.copy(f"{imagesPath}/{image}", f"{path}/2140")
        except Exception as e:
            print(f"An error occurred: {e}")
    elif 41 <= age <= 60:
        try:
            shutil.copy(f"{imagesPath}/{image}", f"{path}/4160")
        except Exception as e:
            print(f"An error occurred: {e}")
    elif 61 <= age <= 80:
        try:
            shutil.copy(f"{imagesPath}/{image}", f"{path}/6180")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        try:
            shutil.copy(f"{imagesPath}/{image}", f"{path}/81100")
        except Exception as e:
            print(f"An error occurred: {e}")