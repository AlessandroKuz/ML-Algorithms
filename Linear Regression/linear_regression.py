import math


def compute_linear_function(x, w, b) -> float:
    """
    Computes the linear function w*x + b
    Args:
        x (scalar): Data
        w,b (scalar): model parameters
    Returns
        f_wb (scalar): The linear function w*x + b
    """

    return w*x + b


def compute_cost(x, y, w, b) -> float:
    """
    Computes the cost for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      cost (scalar): The cost for linear regression 
    """

    # Number of training examples
    m = x.shape[0]
    cost = 0

    # Compute the cost function
    for i in range(m):
        f_wb = compute_linear_function(x[i], w, b)
        cost += (f_wb - y[i])**2
    cost /= (2*m)

    return cost


def compute_gradient_descent(x, y, w, b) -> tuple[float, float]:
    """
    Computes the gradient of the cost function for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dw, db (tuple(float, float)): The gradient of the cost function for linear regression 
    """

    # Number of training examples
    m = x.shape[0]
    dw = 0
    db = 0

    # Compute the gradient of the cost function
    for i in range(m):
        f_wb = compute_linear_function(x[i], w, b)
        dw += (f_wb - y[i])*x[i]
        db += (f_wb - y[i])

    dw /= m
    db /= m

    return dw, db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function, *, plot_points = False) -> tuple(float, float, list, list) | None: 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))   : Data, m examples 
      y (ndarray (m,))   : target values
      w_in, b_in (scalar): initial values of model parameters  
      alpha (float)      : Learning rate
      num_iters (int)    : number of iterations to run gradient descent
      cost_function      : function to call to produce cost
      gradient_function  : function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    if plot_points:
      J_history = []
      p_history = []
    
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if plot_points:
            if i<100000:      # prevent resource exhaustion 
                J_history.append( cost_function(x, y, w , b))
                p_history.append([w,b])
            
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters/10) == 0:
                print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                    f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                    f"w: {w: 0.3e}, b:{b: 0.5e}")
 
            return w, b, J_history, p_history  #return w and J,w history for graphing
