import numpy as np
from numba import njit

@njit
def recursive(duration, J_prev, vol_prev, omega, alpha, beta):
    """
    GARCH Recursive function to forecast the index process.
    
    .. math::
    
        \sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}
    
    For the EWMA process, the parameters become:
    omega = 0
    alpha = 1-lambda
    beta = lambda
    
    .. math::
        
        \sigma^2_t = (1-\lambda) \epsilon^2_{t-1} + \lambda \sigma^2_{t-1}
    """
    print(omega, alpha, beta)
    
    vol_values = []
    for t in range(duration):
        vol_next = omega + alpha*J_prev**2 + beta*vol_prev
        vol_values.append(vol_next)
        vol_prev = vol_next
    return vol_values