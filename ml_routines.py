import numpy as np

# converts image to vector, from m X l X b X 3 to n X m, where n: features, m:training examples 
def image_to_vector(feature_X):
    return feature_X.reshape( feature_X.shape[0], -1).T 

# for normalizing the features.We divide each feature by the norm
# the matrix is in n X m form, so we normalize each feature by the norm of that feature for all examples
def normalize_features( feature_X):
    norm_x = np.linalg.norm(feature_X, axis = 1, ord = 2, keepdims=True)

    return feature_X / norm_x


# for computing the softmax 
def softmax( feature_X ):
    # find the numerator for softmax
    exp_x = np.exp(feature_X)
    # find the denominator for softmax, we take sum of features for each feature
    sum_x = np.sum(feature_X, axis = 1, keepdims=True)

    return exp_x / sum_x

# compute L1 loss function
def compute_L1_loss(yhat, train_y):
    return np.sum( np.abs(train_y - yhat), axis = 1)

# compute L2 loss function
def compute_L2_loss(yhat, train_y):
    return np.sum( np.dot( (yhat - train_y), (yhat - train_y)))



