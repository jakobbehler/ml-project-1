
import numpy as np

# STILL TODO
'''
Logistic regression using gradient descent (y∈ {0,1})
logistic regression(y, tx, initial w,max iters, gamma)

Regularized logistic regression using gradient descent (y∈ {0,1}, with regularization term λ∥w∥2
reg logistic regression(y, tx, lambda , initial w, max iters, gamma)
'''


def compute_MSE_loss(y, X, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        X: numpy array of shape=(N,d)
        w: numpy array of shape=(d,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    N = y.shape[0]
    e = y- X.dot(w)
    Loss = np.dot(e.T, e)/(2*N)
    
    return Loss

# Linear regression using gradient descent ------------------------------------------

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
  
    N = y.shape[0]
    e = y- tx.dot(w)
    delta_L = (-1/N) * tx.T @ e
    
    return delta_L


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
            for each iteration of GD (as well as the final weights)
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y, tx, w)
        loss = compute_MSE_loss(y, tx, w)
        w = w - gamma * gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws

# Linear regression using stochastic gradient descent ----------------------------------

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    B = y.shape[0]
    e = y- tx.dot(w)
    delta_L = (-1/B) * tx.T @ e
    
    return delta_L
    

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        # create random batch of size batch_size
        N = y.shape[0]
        idx = np.random.choice(N, size=batch_size, replace=False)  # random indices
        X_batch = tx[idx]
        y_batch = y[idx]

        # compute gradient
        gradient = compute_stoch_gradient(y_batch, X_batch, w)
        loss = compute_MSE_loss(y_batch, X_batch, w)

        w = w - gamma * gradient
        
        losses.append(loss)
        ws.append(w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws



# Least squares regression using normal equations ------------------------------


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    X = tx.copy() # for better understanding of variable names

    Gram = X.T @ X
    b = X.T @ y 

    # sanity check: if Gram Matrix is invertible!
    rank = np.linalg.matrix_rank(Gram)
    D = Gram.shape[0]

    if rank == D: # invertible iff rank = D
        w = np.linalg.solve(Gram, b)

    else:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)

    # MSE Loss
    N = y.shape[0]
    e = y- X.dot(w)
    mse = np.dot(e.T, e)/(2*N)

    return w, mse

# Ridge regression using normal equations -----------------------------------------------

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    
    N, D = tx.shape

    I = np.identity(D)                     # (D,D)
    Gram_ridge = tx.T @ tx + 2 * N * lambda_ * I
    b = tx.T @ y

    # if lambda = 0 then the gram matrix could be non-invertible!
    if lambda_ > 0:
        w = np.linalg.solve(Gram_ridge, b)
    else: 
        w, *_ = np.linalg.lstsq(tx, y, rcond=None) # backup
        
    return w