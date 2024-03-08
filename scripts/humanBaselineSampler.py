import os, shutil, random

folder_path = '../Cleanup3/UTKFaceCleanup3'  # path to your UTKFace folder

people = []
for filename in os.listdir(folder_path):
    people.append(filename)

sampled = random.sample(people, 200)
random.shuffle(sampled)

output_path = '../Cleanup3/Sample'
for person in sampled:
    shutil.copy(f"{folder_path}/{person}", output_path)

tmp = os.listdir(output_path)
random.shuffle(tmp)

for i, filename in enumerate(tmp):
    try:
        os.rename(f"{output_path}/{filename}", f"{i}_{filename}")
    except OSError as e:
        print(f"Error renaming")