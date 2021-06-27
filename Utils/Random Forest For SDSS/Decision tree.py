# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# Importing the dataset
data = np.load('F:\Online Courses\Data-driven Astronomy\WEEK 5\Python practice\sdss_galaxy_colors.npy')

features = np.zeros(shape=(len(data), 4))
features[:, 0] = data['u'] - data['g']
features[:, 1] = data['g'] - data['r']
features[:, 2] = data['r'] - data['i']
features[:, 3] = data['i'] - data['z']
targets = data['redshift']
#Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, targets_train, targets_test = train_test_split(features,targets,test_size = 0.2,  random_state = 0)
# initialize model
regressor = DecisionTreeRegressor(max_depth=(19),random_state = 0)
regressor.fit(features_train, targets_train)
# get the predicted_redshifts
y_pred = regressor.predict(features_test)  
  





#Write a function that calculates the median of the differences between our predicted and actual values
def median_diff(predicted, actual):
  return np.median(np.abs(y_pred[:] - targets_test[:]))
diff = median_diff(y_pred, targets_test)
print("Median difference: {:0.3f}".format(diff))

#plot the results to see how well our model looks
plt.scatter(targets_test, y_pred, s=0.4)
plt.xlim((0, targets_test.max()))
plt.ylim((0, y_pred.max()))
plt.xlabel('Measured Redshift')
plt.ylabel('Predicted Redshift')
plt.savefig('treeResult', dpi=1200)
plt.show()