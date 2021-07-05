import random

difficulties = ["Easy", "Medium", "Hard"]

choice = random.choices(difficulties, weights=[50, 30, 20])[0]

print(choice)
