import numpy as np
from matplotlib import pyplot as plt

# Complete the following to make the plot
if __name__ == "__main__":
    data = np.load("data/sdss_galaxy_450000.npy")
    # Get a colour map
    cmap = plt.get_cmap("YlOrRd")

    # Define our colour indexes u-g and r-i
    u_g = data["u"] - data["g"]
    r_i = data["r"] - data["i"]

    # Make a redshift array
    redshift = data["redshift"]

    # Create the plot with plt.scatter
    plot = plt.scatter(u_g, r_i, s=0.5, lw=0, c=redshift, cmap=cmap)

    cb = plt.colorbar(plot)
    cb.set_label("Redshift")

    # Define your axis labels and plot title
    plt.clim(0, 3.0)
    plt.xlabel("Colour index  u-g")
    plt.ylabel("Colour index  r-i")
    plt.title("Redshift (colour) u-g versus r-i")

    # Set any axis limits
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1)
    plt.savefig("output/plot/Contourmap", dpi=600)
    plt.show()
