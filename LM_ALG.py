import numpy as np

def levenberg_marquardt_backpropagation(X, Y, hidden_layer_sizes, activation='tanh', lamda=0.01, tol=1e-6, max_iter=100):
    """
    Implementation of the Levenberg-Marquardt algorithm for backpropagation in a neural network.

    Parameters
    ----------
    X : numpy.ndarray
        An array of input data with shape (n_samples, n_features).
    Y : numpy.ndarray
        An array of target values with shape (n_samples, n_outputs).
    hidden_layer_sizes : tuple
        A tuple of integers representing the number of neurons in each hidden layer.
    activation : str, optional
        The activation function to use in the hidden layers (default is 'tanh').
    lamda : float, optional
        The initial value of the LM parameter (default is 0.01).
    tol : float, optional
        The convergence tolerance (default is 1e-6).
    max_iter : int, optional
        The maximum number of iterations (default is 100).

    Returns
    -------
    numpy.ndarray
        The trained weights of the neural network.
    """

    # Define activation function and its derivative
    if activation == 'tanh':
        activation_func = np.tanh
        activation_deriv = lambda x: 1.0 - np.tanh(x) ** 2
    elif activation == 'sigmoid':
        activation_func = lambda x: 1.0 / (1.0 + np.exp(-x))
        activation_deriv = lambda x: activation_func(x) * (1.0 - activation_func(x))
    else:
        raise ValueError('Activation function not supported.')

    # Initialize variables
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_layers = len(hidden_layer_sizes) + 1
    sizes = (n_features,) + hidden_layer_sizes + (n_outputs,)
    weights = [np.random.randn(sizes[i], sizes[i+1]) for i in range(n_layers)]
    prev_weights = [w.copy() for w in weights]
    lamda_vals = [lamda] * n_layers
    iter_num = 0

    # Iterate until convergence or max iterations reached
    while iter_num < max_iter:
        # Compute gradients using backpropagation
        a = [X]
        z = []
        for i in range(n_layers):
            z.append(np.dot(a[i], weights[i]))
            a.append(activation_func(z[-1]))
        delta = [a[-1] - Y]
        for i in range(n_layers-1, 0, -1):
            delta.insert(0, np.dot(delta[0], weights[i].T) * activation_deriv(z[i-1]))

        # Compute the LM parameter for each layer
        for i in range(n_layers):
            diag_h = np.sum(np.square(weights[i]), axis=0)
            lamda_vals[i] = lamda * np.max(diag_h)
            lamda_vals[i] = np.diag(lamda_vals[i])

        # Update the weights
        for i in range(n_layers):
            dw = np.dot(a[i].T, delta[i])
            dw += np.dot(lamda_vals[i], weights[i])
            weights[i] -= np.linalg.solve(dw, np.dot(a[i].T, delta[i]))

        # Check for convergence
        norm_diff = np.linalg.norm([w - pw for w, pw in zip(weights, prev_weights)])
        if norm_diff < tol:
            break
        else:
            prev_weights = [w.copy() for w in weights]
            iter_num += 1

    return weights
