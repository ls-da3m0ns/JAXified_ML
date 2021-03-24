import jax.numpy as jnp
from jax import jit, random, grad
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)


def LeNet5(num_classes):
    return stax.serial(
        
    )