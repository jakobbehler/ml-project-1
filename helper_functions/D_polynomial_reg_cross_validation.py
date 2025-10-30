
import numpy as np

## K indices lists for K FOLD

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    # split automatically handles uneven divisions safely
    k_indices = np.array_split(indices, k_fold)
    return np.array(k_indices, dtype=object)

# ----------------------------------------------------------------------

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """

    # polynomial basis function:
    N = x.shape[0]

    Theta_polynomial_matrix = np.zeros((N, degree + 1))
    Theta_polynomial_matrix[:, 0] = 1

    for i in range(degree):
        for j in range(N):
            Theta_polynomial_matrix[j, i+1] = x[j] ** (i+1)

    return Theta_polynomial_matrix

# ----------------------------------------------------------------------

def build_poly_multidimentsional(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """

    # polynomial basis function:
    N = x.shape[0]

    Theta_polynomial_matrix = np.zeros((N, degree + 1))
    Theta_polynomial_matrix[:, 0] = 1

    for i in range(degree):
        for j in range(N):
            Theta_polynomial_matrix[j, i+1] = x[j] ** (i+1)

    return Theta_polynomial_matrix

# ----------------------------------------------------------------------
# RIDGE REGRESSION
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

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

# ----------------------------------------------------------------------


def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """

    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

# ----------------------------------------------------------------------

## CLASSIC CROSS VALIDATION FOR POLYNOMIAL RIDGE REGRESSION

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    te_idx = k_indices[k, :]                     # test indices
    tr_idx = np.setdiff1d(np.arange(len(y)), te_idx)  # everything else = train

    x_tr = x[tr_idx]
    x_te = x[te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]

 

    x_tr_poly = build_poly(x_tr, degree)
    x_te_poly = build_poly(x_te, degree)

    w = ridge_regression(y_tr, x_tr_poly, lambda_)
    
    loss_tr = np.sqrt(2 * compute_mse(y_tr, x_tr_poly, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, x_te_poly, w))

    return loss_tr, loss_te

# ----------------------------------------------------------------------
## FIND BEST DEGREE AND LAMBDA FOR POLYNOMIAL REGRESSION USING KFOLD

def best_degree_lambda_selection(degrees, k_fold, lambdas, seed=1):
    """cross validation over regularisation parameter lambda and degree.

    Args:
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)

    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_rmse = None
    best_lambda = None
    best_degree = None

    for lambda_ in lambdas:
        for degree in degrees:

            rmse_te_inside = []
            rmse_tr_inside = [] 
            
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_inside.append(loss_te)
                rmse_tr_inside.append(loss_tr)
            
            average_rmse_te = np.mean(rmse_te_inside)
            average_rmse_tr = np.mean(rmse_tr_inside)

            if (best_rmse is None) or (average_rmse_te < best_rmse):
                        best_rmse = average_rmse_te
                        best_lambda = lambda_   
                        best_degree = degree

    return best_degree, best_lambda, best_rmse