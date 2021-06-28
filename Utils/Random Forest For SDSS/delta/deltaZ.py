from DecisionTree import main_tree
from RandomForest import main_forest, plot_forest
import numpy as np
from matplotlib import pyplot as plt

data = np.load("sdss_galaxy_colors.npy")

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
