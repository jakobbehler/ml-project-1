import numpy as np
import helper_functions.B_regression_helpers as regresHelp

## ------------------------------------------------------------------------------------------------
# LINEAR REGRESSION
##------------------------------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final weight vector of shape (D, 1)
        loss: final loss value (scalar)
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    w_final = initial_w
    loss_final = None

    for n_iter in range(max_iters):

        gradient = regresHelp.compute_gradient(y, tx, w)
        loss = regresHelp.compute_MSE_loss(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        w_final = w
        loss_final = loss

    return w_final, loss_final


# Linear regression using stochastic gradient descent ----------------------------------


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final weight vector of shape (D, 1)
        loss: final loss value (scalar)
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        # create random batch of size batch_size (in this case always 1)
        N = y.shape[0]
        idx = np.random.choice(N, size=1, replace=False)  # random indices
        X_batch = tx[idx]
        y_batch = y[idx]

        # compute gradient
        gradient = regresHelp.compute_stoch_gradient(y_batch, X_batch, w)
        loss = regresHelp.compute_MSE_loss(y_batch, X_batch, w)
        w = w - gamma * gradient
        losses.append(loss)
        ws.append(w)

    w = ws[-1]
    loss = losses[-1]

    return w, loss


## ------------------------------------------------------------------------------------------------
# LEAST SQUARES & RIDGE
##------------------------------------------------------------------------------------------------


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: final weight vector of shape (D, 1)
        loss: final loss value (scalar)

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    X = tx.copy()  # for better understanding of variable names

    Gram = X.T @ X
    b = X.T @ y

    # sanity check: if Gram Matrix is invertible!
    rank = np.linalg.matrix_rank(Gram)
    D = Gram.shape[0]

    if rank == D:  # invertible iff rank = D
        w = np.linalg.solve(Gram, b)

    else:  # handle singalrar matrix
        w = np.linalg.pinv(Gram) @ b  # pseudo inverse for singular matrices

    # MSE Loss
    N = y.shape[0]
    e = y - X.dot(w)
    loss = np.dot(e.T, e) / (2 * N)

    return w, loss


# Ridge regression using normal equations -----------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: final weight vector of shape (D, 1)
        loss: final loss value (scalar)

    """

    N, D = tx.shape
    I = np.identity(D)

    # Compute the ridge solution
    Gram_ridge = tx.T @ tx + 2 * N * lambda_ * I
    b = tx.T @ y

    # Handle possible singular matrix
    rank = np.linalg.matrix_rank(Gram_ridge)
    if rank != D:
        w = np.linalg.pinv(Gram_ridge) @ b  # pseudo inverse for singular matrices
    else:
        w = np.linalg.solve(Gram_ridge, b)

    # Compute MSE loss (without penalty term)
    e = y - tx @ w
    loss = (e.T @ e) / (2 * N)

    return w, loss


## ------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION
##------------------------------------------------------------------------------------------------


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
        w: final weight vector of shape (D, s1)
        loss: final loss value (scalar)

    """
    w = initial_w.copy()
    for _ in range(max_iters):
        loss, w = regresHelp.logistic_regression_gradient_descent_step(y, tx, w, gamma)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: initial weights, shape=(D, 1)
        max_iters: number of iterations
        gamma: learning rate

    Returns:
        w: final weight vector of shape (D, 1)
        loss: final loss value (scalar)
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        loss, w = regresHelp.penalized_logistic_regression_gradient_decent_step(
            y, tx, w, gamma, lambda_
        )
    loss = regresHelp.calculate_logistic_loss(
        y, tx, w
    )  # Compute loss (without penalty term)

    return w, loss
