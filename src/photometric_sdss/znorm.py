import pandas as pd
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy.optimize import curve_fit

z=[]
z_spec=[]
znorm=[]

spec=pd.read_excel("data/n=600/spec.xlsx")
photo=pd.read_excel("data/n=600/photo.xlsx")

for i in range(89880):
   z.append(spec.iloc[i,0]-photo.iloc[i,0])
   z_spec.append(spec.iloc[i,0])
   znorm.append(z[i]/(z_spec[i]+1))
deltazaverage=np.mean(znorm)
print(deltazaverage)
dfOut = pd.DataFrame(znorm)
dfOut.to_excel("znorm.xlsx", index=False)



size, scale = 1000, 10
df=pd.read_excel("znorm.xlsx")

ax=df['znorm'].plot.hist(grid=False, bins=200, rwidth=0.9,
                   color='#607c8e')

plt.xlim(-0.5, 0.5)
list1=[-0.5,0.5]
width = 1/1.5
plt.title("\u0394 z distribution")
plt.xlabel('\u0394 z')
plt.ylabel('Number')
# ax.set_yscale('log')
# plt.grid(axis='y', alpha=0.75)
plt.savefig("deltaznorm", dpi=1200)
plt.show()


# fit bar plot data using curve_fit
def func(x, a, b, c):
    # a Gaussian distribution
    return a * np.exp(-(x-b)**2/(2*c**2))

popt, pcov = curve_fit(func, list1, znorm)

x = np.linspace(-0.5, 0.5, 100)
y = func(x, *popt)

plt.plot(x + width/2, y, c='g')