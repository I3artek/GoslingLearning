import os, random

folder_path = '../Cleanup4/Data'

files = {}
for age in range(1, 101):
    files[age] = []

for filename in os.listdir(folder_path):
    age = int(filename.split('_')[0])
    files[age].append(filename)

for age in range(1, 101):
    if len(files[age]) > 500:
        diff = len(files[age]) - 500
        to_delete = random.sample(files[age], k=diff)
        for file in to_delete:
            os.remove(f"{folder_path}/{file}")