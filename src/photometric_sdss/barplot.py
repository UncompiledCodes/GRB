import pandas as pd
import matplotlib.pyplot as plt
import pylab

size, scale = 1000, 10
df=pd.read_csv("data/sdss_galaxy_450000.csv")

ax=df['redshift'].plot.hist(grid=False, bins=200, rwidth=0.9,
                   color='#607c8e')

plt.xlim(0, 7)


plt.title('Redshift Distribution of Galaxies')
plt.xlabel('Redshift')
plt.ylabel('Galaxies')
ax.set_yscale('log')
# plt.grid(axis='y', alpha=0.75)
plt.savefig("Gal_distr", dpi=1200)