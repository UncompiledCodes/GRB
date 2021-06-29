# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = np.load("sdss_galaxy_colors.npy")

features = np.zeros(shape=(len(data), 4))
features[:, 0] = data["u"] - data["g"]
features[:, 1] = data["g"] - data["r"]
features[:, 2] = data["r"] - data["i"]
features[:, 3] = data["i"] - data["z"]
targets = data["redshift"]
# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2, random_state=0
)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=500,max_depth=(16),random_state=0)
regressor.fit(features_train, targets_train)

# Predicting a new result
from scipy.sparse import csr_matrix

# A = csr_matrix([[features[:]]])
# x=np.array(0.31476,0.0571,0.28991,0.07192)
y_pred = regressor.predict(features_test)

#K-fold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = features_train, y = targets_train, cv = 10)
print("Accuracy: {} %".format(accuracies.mean()*100))
print("Standard Deviation: {} %".format(accuracies.std()*100))

#R2
from sklearn.metrics import r2_score

M1_tst_unorm_R2 = r2_score(targets_test, y_pred)
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [16,17,18,19],
#     'n_estimators': [400,500]
# }

# Instantiate the grid search model
# grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, 
#                           cv = 10, n_jobs = -1, verbose = 2)
# grid_search = grid_search.fit(features_train, targets_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)
# Write a function that calculates the median of the differences between our predicted and actual values
def median_diff(predicted, actual):
    return np.median(np.abs(y_pred[:] - targets_test[:]))


diff = median_diff(y_pred, targets_test)
print("Median difference: {:0.3f}".format(diff))

# plot the results to see how well our model looks
plt.scatter(targets_test, y_pred, s=0.4)
plt.xlim((0, targets_test.max()))
plt.ylim((0, y_pred.max()))
plt.xlabel("Measured Redshift")
plt.ylabel("Predicted Redshift")
plt.savefig("RF-Result", dpi=1200)
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
"""X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""


def main_forest():
    return y_pred - targets_test
