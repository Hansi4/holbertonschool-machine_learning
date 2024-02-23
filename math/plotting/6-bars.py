#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here

people = (
    "Farrah",
    "Fred",
    "Felicia",
)
fruit_counts = {
    "apples": np.array([70, 31, 58]),
    "bananas": np.array([82, 37, 66]),
    "oranges": np.array([94, 43, 74]),
    "peaches": np.array([106, 49, 82]),
}
width = 0.5
index = np.arange(len(people))

fig, ax = plt.subplots()
bottom = np.zeros(3)

for boolean, fruit_count in fruit_counts.items():
    p = ax.bar(people, fruit_count, width, label=boolean, bottom=bottom)
    bottom += fruit_count

ax.set_ylabel('Quantity of Fruit')
ax.set_xticks(index)
ax.set_xticklabels(people)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_title("Number of Fruit per Person")
ax.legend(loc="upper right")

plt.show()