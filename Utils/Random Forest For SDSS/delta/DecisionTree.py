# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Importing the dataset


def dtree(data):
    features = np.zeros(shape=(len(data), 4))
    features[:, 0] = data["u"] - data["g"]
    features[:, 1] = data["g"] - data["r"]
    features[:, 2] = data["r"] - data["i"]
    features[:, 3] = data["i"] - data["z"]
    targets = data["redshift"]
    # Splitting the dataset into the Training and Test set

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=0
    )
    # initialize model
    regressor = DecisionTreeRegressor(max_depth=(19), random_state=0)
    regressor.fit(features_train, targets_train)
    # get the predicted_redshifts
    y_pred = regressor.predict(features_test)
    
    accuracies = cross_val_score(estimator = regressor, X = features_train, y = targets_train, cv = 10)

    return [y_pred, targets_test, accuracies]


def median_diff(predicted, actual):
    return np.median(np.abs(predicted[:] - actual[:]))


def plot_tree(data):
    y_pred, targets_test = dtree(data)
    plt.scatter(targets_test, y_pred, s=0.4)
    plt.xlim((0, targets_test.max()))
    plt.ylim((0, y_pred.max()))
    plt.xlabel("Measured Redshift")
    plt.ylabel("Predicted Redshift")
    plt.savefig("plot/Tree_Result", dpi=1200)
    plt.show()


def main_tree(data):
    y_pred, targets_test, accuracies = dtree(data)
    diff = median_diff(y_pred, targets_test)
    print(f"Median difference of decision tree: {diff}")
    print("Accuracy decision tree: {} %".format(accuracies.mean()*100))
    print("Standard Deviation decision tree: {} %".format(accuracies.std()*100))
    delta_tree = y_pred - targets_test
    return delta_tree
