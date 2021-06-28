from DecisionTree import main_tree
from RandomForest import main_forest, plot_forest
import numpy as np
from matplotlib import pyplot as plt
import sys


def main(data):
    tree = main_tree(data)
    forest = main_forest(data)
    plt.scatter(tree, forest, s=0.4)
    plt.xlim((0, tree.max()))
    plt.ylim((0, forest.max()))
    plt.xlabel("Tree delta Z")
    plt.ylabel("Forest delta Z")
    plt.savefig("deltaZ", dpi=1200)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


if __name__ == "__main__":
    try:
        data = np.load(sys.argv[1])
    except:
        data = np.load("sdss_galaxy_colors.npy")
    main(data)
