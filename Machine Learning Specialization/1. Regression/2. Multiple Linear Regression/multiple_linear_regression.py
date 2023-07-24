import numpy as np
import copy
import math


def predict_single_loop(x, w, b) -> float: 
    """
    Single prediction using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar) : Model parameter     
      
    Returns:
      p (scalar) : Prediction
    """

    # # Ineficient way
    # n = x.shape[0]
    # p = 0

    # for i in range(n):
    #     p_i = x[i] * w[i]  
    #     p += p_i  
    # p += b     

    # Efficient way
    p = np.dot(x, w) + b

    return p


def compute_cost(X, y, w, b) -> float: 
    """
    Compute cost

    Args:
      X (ndarray (m, n)): Data, m examples with n features
      y (ndarray (m,))  : Target values
      w (ndarray (n,))  : Model parameters  
      b (scalar)        : Model parameter
    
    Returns:
      cost (scalar)     : Cost
    """

    m = X.shape[0]
    cost = 0.

    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           # (n,)(n,) = scalar
        cost += (f_wb_i - y[i])**2             # scalar
    cost /= (2 * m)                            # scalar    

    return cost


def compute_gradient(X, y, w, b) -> tuple: 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m, n)): Data, m examples with n features
      y (ndarray (m,))  : Target values
      w (ndarray (n,))  : Model parameters  
      b (scalar)        : Model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    
    m,n = X.shape           # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] += err * X[i, j]    
        dj_db += err                        
    
    dj_dw /= m                                
    dj_db /= m                                
        
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, *, plot_points=False): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m, n))  : Data, m examples with n features
      y (ndarray (m,))    : Target values
      w_in (ndarray (n,)) : Initial model parameters  
      b_in (scalar)       : Initial model parameter
      cost_function       : Function to compute cost
      gradient_function   : Function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : Number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
    """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    if plot_points:
        J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None

        if plot_points:
            # Save cost J at each iteration
            if i < 100000:      # prevent resource exhaustion 
                J_history.append( cost_function(X, y, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
        return w, b, J_history  # return final w,b and J history for graphing
    
    return w, b


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


def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw
