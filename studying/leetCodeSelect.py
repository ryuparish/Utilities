import random

difficulties = ["Easy", "Medium", "Hard"]
language = ["Java", "C++", "Python"]

#difficulty_choice = random.choices(difficulties, weights=[40, 35, 25])[0]
difficulty_choice = random.choices(difficulties, weights=[30, 40, 30])[0]
language_choice = random.choices(language, weights=[70, 20, 10])[0]

print(difficulty_choice, "in", language_choice)
