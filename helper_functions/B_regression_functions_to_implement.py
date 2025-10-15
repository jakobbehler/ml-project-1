
import numpy as np

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


## ------------------------------------------------------------------------------------------------
#. LOGISTIC REGRESSION
###------------------------------------------------------------------------------------------------

'''
Logistic regression using gradient descent (y∈ {0,1})
logistic regression(y, tx, initial w,max iters, gamma)
'''

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    
    n = y.shape[0]
    loss = 0

    for i in range(n):
        sig = sigmoid(tx[i].T @ w)
     
        loss += y[i] * np.log(sig) + (1-y[i]) * np.log(1-sig)
    
    return (-loss/n).item() # for some reason without item() its treated as a 1x1 numpy array, not a scalar


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    N = y.shape[0]

    Xw_sig = np.zeros((N, 1)) # X(N,D) @ w(D,1) = Xw (N,1)

    Xw = tx @ w # (N,1) vector

    for i in range(N):
        Xw_sig[i] = sigmoid(Xw[i]) # because sigmoid works on scalars, not vectors we apply it feature by feature 

    return tx.T @ (Xw_sig-y) / N


def logistic_regression_gradient_descent_step(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    
    w_prime = w - gamma * calculate_logistic_gradient(y, tx, w)
    loss = calculate_logistic_loss(y, tx, w)
    return loss, w_prime

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: initial weights, shape=(D, 1)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: final weights (D, 1)
        weights: list of weight vectors per iteration
        losses: list of loss values per iteration
    """
    losses = []
    weights = []

    w = initial_w.copy()

    for iter in range(max_iters):
        loss, w_new = logistic_regression_gradient_descent_step(y, tx, w, gamma)
        losses.append(loss)
        weights.append(w_new.copy()) 
        w = w_new

    return w, weights, losses

'''
Regularized logistic regression using gradient descent (y∈ {0,1}, with regularization term λ∥w∥2
reg logistic regression(y, tx, lambda , initial w, max iters, gamma)
'''

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
 
    loss = calculate_logistic_loss(y, tx, w)
    gradient = calculate_logistic_gradient(y, tx, w)

    gradient = gradient + 2 * lambda_ * w

    return loss, gradient


def penalized_logistic_regression_gradient_decent_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
  
    w_t = w - gamma * gradient
    return loss, w_t



def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: initial weights, shape=(D, 1)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: final weights (D, 1)
        weights: list of weight vectors per iteration
        losses: list of loss values per iteration
    """
    losses = []
    weights = []

    w = initial_w.copy()

    for iter in range(max_iters):
        loss, w_new = penalized_logistic_regression_gradient_decent_step(y, tx, w, gamma, lambda_)
        losses.append(loss)
        weights.append(w_new.copy()) 
        w = w_new

    return w, weights, losses
