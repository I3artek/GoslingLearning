import os, random

folder_path = '../Cleanup3/UTKFaceCleanup3'
ones = []
twentysix = []
for filename in os.listdir(folder_path):
    age = int(filename.split('_')[0])
    if age == 1:
        ones.append(filename)
    elif age == 26:
        twentysix.append(filename)

one_delete = random.sample(ones, k=len(ones) // 2)
twentysix_delete = random.sample(twentysix, k=len(twentysix) // 2)

joined = one_delete + twentysix_delete
for file in joined:
    os.remove(f"{folder_path}/{file}")