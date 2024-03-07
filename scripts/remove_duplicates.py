import cv2
import os
from skimage import metrics
# Load images

path = '../UTKFace'
dupes = []

    for i, filename1 in enumerate(os.listdir(path)):
        for j, filename2 in enumerate(os.listdir(path)):
            if filename2 <= filename1:
                continue
            age1 = int(filename1.split('_')[0])
            age2 = int(filename2.split('_')[0])
            if abs(age2 - age1) > 20:
                continue
            image1 = cv2.imread(f"{path}/{filename1}")
            image2 = cv2.imread(f"{path}/{filename2}")
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
            score = ssim_score[0]
            #print(f"SSIM Score: ", round(ssim_score[0], 2))

            if(score >= 0.7):
                print("found")
                output.write(f"{filename1} {filename2} {i} {j}\n")
                output.flush()