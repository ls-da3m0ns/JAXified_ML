from jax.typing import ArrayLike
import jax.numpy as np 

def mse(y_pred: ArrayLike, y_true: ArrayLike) -> ArrayLike:
    """
    Mean Squared Error.
    np.sum( 
        np.power( (y_pred - y_true), 2 )
    ) / len(y_true)
    """
    return np.sum( np.power( (y_pred - y_true), 2 ) ) / len(y_true)


    
    
