import pandas as pd
import matplotlib.pyplot as plt


size, scale = 1000, 10
df = pd.read_csv("./data/znormdata.csv")

ax = df["0"].plot.hist(grid=False, bins=200, rwidth=0.9, color="darkred")

plt.xlim(-0.6, 0.6)


plt.title("\u0394 z distribution")
plt.xlabel("\u0394 z")
plt.ylabel("Number")
# ax.set_yscale("log")
# plt.grid(axis='y', alpha=0.75)
plt.savefig("deltaz_under2", dpi=1200)
