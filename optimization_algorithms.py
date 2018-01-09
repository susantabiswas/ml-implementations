def gradient_descent(parameters, train_X, train_y, y_hat, learning_rate):
    '''
    for doing gradient descent and updating the weights
    
    Arguments:
        parameters: (dict) contains the learned weight and bias parameters
        train_X:(numpy matrix) contains the input features for m training examples
        train_y:(numpy matrix) contains the correct output labels for m training examples
        y_hat:(numpy matrix) contains the sigmoid output values for m training examples
        learning_rate: (float) learning rate
    Returns:
        parameters: (dict) contains the learned weight and bias parameters
    '''
    # find the derivatives
    dw = (1 / train_X.shape[1]) * np.dot(train_X, (y_hat - train_y).T)
    db = (1 / train_X.shape[1]) * np.sum(y_hat - train_y)

    # update the parameters
    parameters['W'] = parameters['W'] - (learning_rate * dw)
    parameters['b'] = parameters['b'] - (learning_rate * db)

    return parameters
