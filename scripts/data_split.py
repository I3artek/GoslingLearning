import os, random, shutil
path = "../80_10_10/All"
val_path = "../80_10_10/Validation"
test_path = "../80_10_10/Test"

chunk_size = (len(os.listdir(path)) * 10) // 100

validation_sample = random.sample(os.listdir(path), k=chunk_size)

for file in validation_sample:
    shutil.move(f"{path}/{file}", f"{val_path}/{file}")

test_sample = random.sample(os.listdir(path), k=chunk_size)

for file in test_sample:
    shutil.move(f"{path}/{file}", f"{test_path}/{file}")
