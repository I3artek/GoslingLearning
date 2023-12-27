import os
import matplotlib.pyplot as plt
def draw_histogram(data, labelx, labely):
    plt.hist(data, bins=range(min(data), max(data) + 2), align='left', rwidth=0.8)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.xticks(range(min(data), max(data) + 1, 1), rotation='horizontal')
    plt.show()

def main():
    folder_path = '../Group Histograms/70-100'  # path to your UTKFace folder

    # Iterate through files in the folder
    age = []
    gender = []
    race = []
    for filename in os.listdir(folder_path):
        tmp = filename.split('_')
        age.append(int(tmp[0]))
        gender.append(int(tmp[1]))
        race.append(int(tmp[2]))

    counts = [0] * 101
    for person in age:
        counts[person] += 1

    for i in range(1, 101):
        print(f"{i}: {counts[i]}")

    # Draw histogram
    #draw_histogram(age, 'age', 'counts')
    draw_histogram(gender, 'gender', 'counts')
    draw_histogram(race, 'race', 'counts')

if __name__ == "__main__":
    main()
