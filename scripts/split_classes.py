import os, shutil

path = "../80_10_10_augprep/Train_augprep"
files = os.listdir(path)
for i in range (1, 91):
    os.mkdir(f"{path}/{i}")

for filename in files:
    age = filename.split('_')[0]
    if int(age) > 90:
        continue
    shutil.move(f"{path}/{filename}", f"{path}/{age}/{filename}")