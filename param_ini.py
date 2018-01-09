def initialize_parameters(n):
    '''
    for initializing the weight and bias parameters with zero value
    
    Arguments:
        n: no. of input features
    Returns:
        parameters: (dict) zero initialized weight and bias parameters 
    '''
    parameters = {}
    # we will use zero initialization
    parameters['W'] = np.zeros((n, 1), dtype=np.float64)
    # bias
    parameters['b'] = 0

    return parameters
