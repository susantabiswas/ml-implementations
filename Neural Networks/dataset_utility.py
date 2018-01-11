# contains different functions for loading, preprocessing of dataset
import h5py
import numpy as np

# for loading the multi class classification dataset
def load_data1(file_loc):
    '''
    Loads the data set from the data set file
    Arguments:
        file_loc: (str) file location of data set
    returns:
        train_X:(numpy matrix) contains the training features
        train_y:(numpy matrix) contains the correct labels for the training examples
        test_X:(numpy matrix) contains the test features
        test_y:(numpy matrix) contains the correct labels for the test examples
    '''
    test_loc = r'datasets\train_catvsnoncat.h5'
    train_loc = r'datasets\test_catvsnoncat.h5'
    # load the data set
    data = h5py.File(file_loc,"r")
    
    # load the train file
    # load the training features
    train_X = np.array(data['images'][0:180])
    # load the correct labels for training set
    train_y = np.array(data['labels'][0:180])

    # load the test file
    # load the test features
    test_X = np.array(data['images'][180:])
    # load the correct labels for test set
    test_y = np.array(data['labels'][180:])

    return train_X, train_y, test_X, test_y

# for loading the binary class classification dataset
def load_data2(train_loc, test_loc):
    '''
    Loads the data set from the data set file
    Arguments:
        train_loc: (str) file location of training data set
        test_loc: (str) file location of test data set
    returns:
        train_X:(numpy matrix) contains the training features
        train_y:(numpy matrix) contains the correct labels for the training examples
        test_X:(numpy matrix) contains the test features
        test_y:(numpy matrix) contains the correct labels for the test examples
    '''
    test_loc = r'datasets\test_catvsnoncat.h5'
    train_loc = r'datasets\train_catvsnoncat.h5'
    # load the train file
    train_data = h5py.File(train_loc,"r")
    # load the training features
    train_X = np.array(train_data['train_set_x'][:])
    # load the correct labels for training set
    train_y = np.array(train_data['train_set_y'][:])
    train_data.close()
    
    # load the test file
    test_data = h5py.File(test_loc, "r")
    # load the test features
    test_X = np.array(test_data['test_set_x'][:])
    # load the correct labels for test set
    test_y = np.array(test_data['test_set_y'][:])
    test_data.close()
    
    return train_X, train_y, test_X, test_y
    


def normalize_data( train_data, test_data):
    ''' 
    Normalizes the data set
    Arguments:
        train_data: (numpy matrix) training data
        test_data: (numpy matrix) testing data
    Returns:
        train_data_norm: (numpy matrix) normalized training data
        test_data_norm: (numpy matrix) normalized testing data
    '''
    # for images we can normalize them by dividing them with max intensity value of 255
    train_data_norm = train_data / 255
    test_data_norm = test_data / 255

    return train_data_norm, test_data_norm



def unroll_features(train_X, test_X):
    '''
    Unrolls the feature vector into a single array, i.e for put the pixel values for 3 RGB channels in 
    the same array

    Arguments: 
        train_X :(numpy matrix) training data features with dimensions : m X l X b X 3, l: length, b: breadth m: training examples
        test_X :(numpy matrix) testing data features with dimensions : m X l X b X 3, l: length, b: breadth m: training examples
    Returns:
        train_X :(numpy matrix) flattened training data features with dimensions : n X m , n: features, m: training examples
        test_X :(numpy matrix) flattened testing data features with dimensions : n X m , n:features, m: training examples
     
    '''

    # Unroll/Flatten the feature vectors
    train_X = train_X.reshape(train_X.shape[0], -1).T
    test_X = test_X.reshape(test_X.shape[0], -1).T

    return train_X, test_X


def one_hot_matrix( train_y, test_y, c):
    '''
        For creating one hot matrix for output labels
        Arguments:
            train_y :(numpy matrix) training data output labels with dimensions : m X 1 , m: training examples
            test_y :(numpy matrix) testing data output labels with dimensions : m X 1,  m: testing examples
            c: (int) no. of classes
        Returns:
            train_y :(numpy matrix) training data output labels with dimensions : c X m , c: no. of classes, m: training examples
            test_y :(numpy matrix) testing data output labels with dimensions : c X m , c: no. of classes, m: testing examples
        
    '''

    # converting correct labels matrix to one hot matrix
    train_y = np.eye(c)[train_y.reshape(-1)].T
    test_y = np.eye(c)[test_y.reshape(-1)].T

    return train_y, test_y
