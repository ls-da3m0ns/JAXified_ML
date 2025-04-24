import jax.numpy as jnp
from jax import jit, random, grad
from jax.example_libraries import stax
from jax.example_libraries.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)


def LeNet5(num_classes):
    return stax.serial(
        GeneralConv(('HWCN','OIHW','NHWC'), 64, (7,7), (2,2), 'SAME'),
        BatchNorm(),
        Relu,
        AvgPool((3,3)),

        Conv(16, (5,5), strides = (1,1),padding="SAME"),
        BatchNorm(),
        Relu,
        AvgPool((3,3)),

        Flatten,
        Dense(num_classes*10),
        Dense(num_classes*5),
        Dense(num_classes),
        LogSoftmax
    )