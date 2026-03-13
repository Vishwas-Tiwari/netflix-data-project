import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("netflix_titles.csv")

type_count = df['type'].value_counts()

type_count.plot(kind='bar')

plt.title("Movies vs TV Shows")

plt.show()