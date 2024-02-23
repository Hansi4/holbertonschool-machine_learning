#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

fruit = np.array([[10, 15, 7],
                  [5, 8, 12],
                  [12, 10, 5],
                  [7, 10, 15]])

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
persons = ['Farrah', 'Fred', 'Felicia']

# Stacked bar plot
fig, ax = plt.subplots()

for i in range(len(fruit)):
    bottom = np.sum(fruit[:i], axis=0) if i > 0 else 0
    ax.bar(persons, fruit[i], bottom=bottom, label=fruits[i], color=colors[i], width=0.5)

ax.set_ylabel('Quantity of Fruit')
ax.set_title("Number of Fruit per Person")
ax.legend()
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
plt.show()