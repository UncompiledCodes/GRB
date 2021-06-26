from sklearn import tree
from KFoldCrossValidatedTree import main_tree
from RandomForestForSDSS import main_forest
from matplotlib import pyplot as plt

tree = main_tree()
forest = main_forest()

plt.scatter(tree, forest, s=0.4)
plt.xlim((0, tree.max()))
plt.ylim((0, forest.max()))
plt.xlabel('Tree delta Z')
plt.ylabel('Forest delta Z')
plt.savefig('deltaZ', dpi=1200)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()