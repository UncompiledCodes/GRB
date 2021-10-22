import pandas as pd
import matplotlib.pyplot as plt


size, scale = 1000, 10
df = pd.read_csv("data/output_under2.csv")

ax = df["redshift"].plot.hist(grid=False, bins=200, rwidth=0.9, color="#607c8e")

plt.xlim(0, 2)


plt.title("Redshift Distribution of Galaxies\QSO")
plt.xlabel("Redshift")
plt.ylabel("Galaxies\QSO")
ax.set_yscale("log")
# plt.grid(axis='y', alpha=0.75)
plt.savefig("Gal_distr_under2", dpi=1200)
