import jax.numpy as jnp
from jax import jit, random, grad
from jax.example_libraries import stax
from jax.example_libraries.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)

def convBlock(ks, filters, stride=(1,1)):
    Main = stax.serial(
            Conv(filters[0], (1,1), strides = (1,1)),
            BatchNorm(),
            Relu,

            Conv(filters[1], (ks,ks), strides=stride),
            BatchNorm(),
            Relu,

            Conv(filters[2],(1,1), strides=(1,1)),
            BatchNorm(),
            Relu
    )

    Shortcut = stax.serial(
            Conv(filters[3], (1,1), strides = stride),
            BatchNorm(),
    )

    fullInternal = stax.parallel(Main,Shortcut)

    return stax.serial(FanOut(2),
                       fullInternal,
                       FanInSum,
                       Relu)

def identityBlock(ks,filters):
    def construct_main(inp_shape):
        return stax.serial(
            Conv(filters[0], (1,1), strides = (1,1)),
            BatchNorm(),
            Relu,

            Conv(filters[1], (ks,ks), padding="SAME"),
            BatchNorm(),
            Relu,

            Conv(input_shape[3], (1,1)),
            BatchNorm(),
        )
    Main = stax.shape_dependent(construct_main)
    return stax.serial( FanOut(2),
                        stax.parallel(Main,Identity),
                        FanInSum,
                        Relu
                      )

def ResNet(num_classes):
    return stax.serial(
            GeneralConv(('HWCN','OIHW','NHWC'), 64, (7,7), (2,2), 'SAME'),
            BatchNorm(),
            Relu,
            MaxPool((3,3), strides=(2,2)),

            convBlock(3, [64,64,256]),
            identityBlock(3, [64,64]),
            identityBlock(3,[64,64]),

            convBlock(3,[128,128,512]),
            identityBlock(3,[128,128]),
            identityBlock(3,[128,128]),
            identityBlock(3,[128,128]),

            convBlock(3,[256,256,1024]),
            identityBlock(3,[256,256]),
            identityBlock(3,[256,256]),
            identityBlock(3,[256,256]),
            identityBlock(3,[256,256]),
            identityBlock(3,[256,256]),

            convBlock(3,[512,512,2048]),
            identityBlock(3,[512,512]),
            identityBlock(3,[512,512]),

            AvgPool((7,7)),
            Flatten,
            Dense(num_classes),
            LogSoftmax
    )

