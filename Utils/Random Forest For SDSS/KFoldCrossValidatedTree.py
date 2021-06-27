import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))

# complete this function
def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    model.fit(train_features, train_targets)
        
    # predict using the model
    predictions = model.predict(test_features)
        
    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions



# data = np.load('sdss_galaxy_colors.npy')
# features, targets = get_features_targets(data)

# # initialize model
# dtr = DecisionTreeRegressor(max_depth=19)

# # call your cross validation function
# predictions = cross_validate_predictions(dtr, features, targets, 10)

# # calculate and print the rmsd as a sanity check
# diffs = median_diff(predictions, targets)
# print('Median difference: {:.3f}'.format(diffs))

# plot the results to see how well our model looks
# plt.scatter(targets, predictions, s=0.4)
# plt.xlim((0, targets.max()))
# plt.ylim((0, predictions.max()))
# plt.xlabel('Measured Redshift')
# plt.ylabel('Predicted Redshift')
# plt.show()

def main_tree():
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  predictions = cross_validate_predictions(dtr, features, targets, 10)
  return predictions[:10000]-targets[:10000]