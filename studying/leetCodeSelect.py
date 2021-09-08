import random

difficulties = ["Medium", "Hard"]
subjects = ["Dynamic Programming", "Trees", "Graph", "Sliding Window", "Array", "String", "Hash Table", "Stack", "Union Find", "Bit Manip", "Queues", "Heaps", "Recursion", "Backtracking", "DFS", "BFS"]
language = ["Java", "C++", "Python"]

#difficulty_choice = random.choices(difficulties, weights=[40, 35, 25])[0]
difficulty_choice = random.choices(difficulties, weights=[80, 20])[0]
language_choice = random.choices(language, weights=[70, 20, 10])[0]

print(difficulty_choice, "in", language_choice)
