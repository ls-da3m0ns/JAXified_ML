import jax
import jax.numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike, DTypeLike
from typing import Any, Callable, Optional, Sequence, Tuple, Union, Type
from typing_extensions import Self
from jax import random
from jaxified.utils import mse


class OLS():
    weights: ArrayLike
    num_features = None
    lowest_training_loss = 1e100
    def __init__(self,
                 learning_rate:float = 0.01, 
                 tolerance:float=1e-4,
                 max_steps:float=1000, 
                 learning_rate_decay:float=0.9,
                 init_method:str="random",
                 random_key : Optional[int] = 0,
                 penalty:str="none"):
        
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.init_method = init_method
        self.key = jax.random.PRNGKey(random_key)
        self.learning_rate_decay = learning_rate_decay
        self.penalty = penalty
        
    
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fits OLS model to the data.
        X: ArrayLike : shape (n_samples, n_features)
        y: ArrayLike : shape (n_samples, 1)
        """
        self.num_features = X.shape[1]
        self.weights = self._initialize_weights(self.key, self.init_method, self.num_features) # (n_features+1, 1)
        X_bias = np.hstack( (np.ones(shape=(X.shape[0], 1)), X) ) # shape would be (n_samples, n_features + 1)
        self._train(X_bias,y)
        return self 
    
    @staticmethod
    def _initialize_weights(key,init_method,num_features) -> ArrayLike:
        """
        Initializes weights for OLS.
        """
        
        if init_method == "random":
            return jax.random.normal(key, shape=(num_features + 1, 1) )
        elif init_method == "zeros":
            return np.zeros(shape=(num_features + 1, 1))
        elif init_method == 'ones':
            return np.ones(shape=(num_features+1, 1))
        elif init_method == 'he_normal':
            return jax.random.normal(key, shape=(num_features + 1, 1) ) * np.sqrt(2 / num_features)
        elif init_method == 'he_uniform':
            return jax.random.uniform(key, shape=(num_features + 1, 1) ) * np.sqrt(2 / num_features)
        
    @staticmethod
    def _predict( X_bias: ArrayLike, weights:ArrayLike ) -> ArrayLike:
        """
        Predicts the output for the given input.
        X: ArrayLike : shape (n_samples, n_features)
        """
        return np.dot(X_bias, weights)  # shape eqn : (n_samples, n_features + 1) dot (n_features + 1, 1) = (n_samples, 1)
    
    @staticmethod
    def _cost(X: ArrayLike, y: ArrayLike, weights:ArrayLike , penalty:str = 'none') -> ArrayLike:
        """
        Cost function for OLS.
        Uses Mean Squared Error. for cost
        
        divide actual mse by 2 to make gradient computation easier.
        """
        y_hat = OLS._predict(X, weights)
        mse_cost = mse(y_hat, y) / 2
        
        if penalty == "none":
            penalty_term = 0
        elif penalty == "l1":
            penalty_term = np.sum( weights )
        return mse_cost + penalty_term
    
    def _update_weights(self, X:ArrayLike, y:ArrayLike) -> None:
        """
        Optimize weights using gradient decent
        """
        grad_fn = jax.grad( self._cost, 2)
        gradients = grad_fn(X,y, self.weights) # shape: (n_features+1, 1) 
        #print(gradients)
        self.weights -= self.learning_rate * gradients
        self.learning_rate *= self.learning_rate_decay
    
    def _train(self, X: ArrayLike, y: ArrayLike ) -> None:
        """
        Loops the OLS algorithm until convergence.
        """
        stoped_at_step = self.max_steps
        for loop_idx in range(self.max_steps):
            curr_loss = self._cost(X,y, self.weights)
            if loop_idx == 0 or curr_loss < self.lowest_training_loss:
                self.lowest_training_loss = curr_loss
            if curr_loss < self.tolerance:
                stoped_at_step = loop_idx
                break 
            self._update_weights(X,y)
            
        
        if stoped_at_step == self.max_steps:
            print("Failed to converge !")
    
    def predict(self, X:ArrayLike):
        assert X.shape[1] == self.num_features, "Number of features did not match with training data"
        X_bias = np.hstack( (np.ones(shape=(X.shape[0], 1)), X) ) # shape would be (n_samples, n_features + 1)
        return self._predict(X_bias, self.weights)
        