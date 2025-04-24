import flax
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from jax.nn.initializers import glorot_normal, normal, ones, zeros
import tensorflow_datasets as tfds

class AlexNet(nn.Module):
    def __init__(self):
        print("Initializing AlexNet")
    
    def setup(self,num_classes,custom_head=None):
        self.head = custom_head if custom_head else nn.Dense(num_classes)
        self.conv96 = nn.Conv(features=96, kernel_size = (11,11), stride=(4,4), padding="VALID", kernel_init=glorot_normal())
        self.conv256 = nn.Conv(features=256, kernel_size = (5,5), stride=(1,1), padding="SAME", kernel_init=glorot_normal())
        self.conv384 = nn.Conv(features=384, kernel_size = (3,3), stride=(1,1), padding="SAME", kernel_init=glorot_normal())
        self.conv384_2 = nn.Conv(features=384, kernel_size = (3,3), stride=(1,1), padding="SAME", kernel_init=glorot_normal())
        self.conv256_2 = nn.Conv(features=256, kernel_size = (3,3), stride=(1,1), padding="SAME", kernel_init=glorot_normal())
        self.maxpool = nn.max_pool
        self.flatten = nn.flatten
        self.dropout = nn.dropout
        self.batchnorm = nn.BatchNorm
        self.relu = nn.relu
        self.logsoftmax = nn.log_softmax
        self.dense1 = nn.Dense(features=4096, kernel_init=glorot_normal())
        self.dense2 = nn.Dense(features=4096, kernel_init=glorot_normal())
        self.dense3 = nn.Dense(features=1000, kernel_init=glorot_normal() )

    def __call__(self,x):
        x = self.conv96(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x, window_shape=(3,3), strides=(2,2), padding="VALID")

        x = self.conv256(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x, window_shape=(3,3), strides=(2,2), padding="VALID")

        x = self.conv384(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv384_2(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv256_2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x, window_shape=(3,3), strides=(2,2), padding="VALID")

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x, rate=0.3)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout(x, rate=0.1)
        x = self.dense3(x)
        x = self.head(x)
        x = self.logsoftmax(x)
        return x


### Loading dataset
def load_dataset():
    dataset, info = tfds.load('mnist', with_info=True)
    dataset_numpy = tfds.as_numpy(dataset)
    train_data,test_data = dataset_numpy['train'],dataset_numpy['test']
    train_data = jnp.array([x['image'] for x in train_data])
    train_data = jnp.reshape(train_data,(train_data.shape[0],28,28,1))
    train_data = train_data.astype(jnp.float32)
    train_data = train_data/255.0
    train_labels = jnp.array([x['label'] for x in dataset_numpy['train']])
    train_labels = train_labels.astype(jnp.int32)

    test_data = jnp.array([x['image'] for x in test_data])
    test_data = jnp.reshape(test_data,(test_data.shape[0],28,28,1))
    test_data = test_data.astype(jnp.float32)
    test_data = test_data/255.0
    test_labels = jnp.array([x['label'] for x in dataset_numpy['test']])
    test_labels = test_labels.astype(jnp.int32)
    
    return train_data,train_labels,test_data,test_labels


### Training 
"""
    Steps : 
     * contruct model 
     * initialize model
     * batch the data
     * define optimizer
     * define loss function
     * create a loop of n epoch
     * for each epoch, loop over the batches
     * for each batch, compute the loss and update the model parameters
     * test the model
     * save the model
"""

