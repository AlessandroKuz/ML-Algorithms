"""
in order to regularize the model, we add a regularization term to the cost function.
The regularization term is the sum of the squares of the weights multiplied by a regularization parameter lambda_.
The regularization parameter lambda_ controls the amount of regularization.
If lambda_ is too large, the model will be too constrained and will not fit the data well,
on the other hand if lambda_ is too small, the model will not be regularized and may overfit the training data.
The regularization term does not include the bias term b (this is standard practice, as regulazing b leads to very tiny changes).
The cost function for linear regression with regularization is:
J(w,b) = (1/2m) * sum((f_wb_i - y_i)^2) + (lambda_/(2m)) * sum(w_j^2)
where:
m is the number of training examples
f_wb_i is the linear function w*x_i + b
y_i is the target value for example i
w_j is the jth element of the weight vector w
lambda_ is the regularization parameter

The gradient of the cost function for linear regression with regularization is:
dw_j = (1/m) * sum((f_wb_i - y_i)*x_i) + (lambda_/m) * w_j
db = (1/m) * sum((f_wb_i - y_i))
where:
m is the number of training examples
f_wb_i is the linear function w*x_i + b
y_i is the target value for example i
x_i is the ith element of the input vector x
w_j is the jth element of the weight vector w
lambda_ is the regularization parameter

The gradient descent update rules for linear regression with regularization are:
w_j = w_j - alpha * dw_j
b = b - alpha * db
where:
alpha is the learning rate
dw_j is the gradient of the cost function with respect to w_j
db is the gradient of the cost function with respect to b
"""
import numpy as np


def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2                               #scalar             
    cost = cost / (2 * m)                                              #scalar  
 
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar
