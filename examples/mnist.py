import os,time
import jax.numpy as jnp
from jax import jit,grad,vmap
from jax import random
from jax.scipy.special import logsumexp
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def random_layer_params(m,n,key,scale=1e-2):
    """
    initialize layer with random weights
    n,m are suze while key is random's key
    scale = random variation in each bias
    w_key,b_key are weights_key and bias_key
    """
    w_key,b_key = random.split(key)
    return scale * random.normal(w_key, (n,m)) , scale*random.normal(b_key,(n,))

def init_network_params(sizes,key):
    keys = random.split(key,len(sizes))
    return [random_layer_params(m,n,k) for m,n,k in
            zip(sizes[:-1],sizes[1:],keys)]

def relu(x):
    return jnp.maximum(0,x)

def predict(params,x):
    temp = x

    for w,b in params[:-1]:
        out = jnp.dot(w,temp) + b
        temp = relu(out)

    w_f,b_f = params[-1]
    logits = jnp.dot(w_f,temp) + b_f
    return logits - logsumexp(logits)

batch_predict = vmap(predict, in_axes=(None,0))

def one_hot(x,y,dtype=jnp.float32):
    return jnp.array(x[:,None] == jnp.arange(y), dtype)

def accuracy(params,x,y):
    target = jnp.argmax(y,axis=1)
    pred = jnp.argmax(batch_predict(params,x),axis=1)
    return jnp.mean(target == pred)

def loss(params,x,y):
    pred = batch_predict(params,x)
    return -jnp.mean(pred * y)

@jit
def update(params,x,y):
    grads = grad(loss)(params,x,y)
    return [(w - step_size*dw, b - step_size*db) for (w,b),(dw,db) in
            zip(params,grads)]

def get_train_batches():
    ds = tfds.load(name="mnist", split="train",
                   as_supervised=True,data_dir=data_dir)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)

#params
data_dir = './tmp/'
layers_sizes = [784,512,512,10]
param_scale = 0.1
step_size = 0.01
num_epochs = 10
batch_size  = 128
n_targets = 10

#data steps 
mnist_data,info = tfds.load(name="mnist",batch_size=-1,
                            data_dir=data_dir,with_info = True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'],mnist_data['test']
num_labels = info.features['label'].num_classes
h, w,c = info.features['image'].shape
num_pixels = h*w*c

#train batches
train_images,train_labels = train_data['image'],train_data['label']
train_images = jnp.reshape(train_images,(len(train_images),num_pixels))
train_labels = one_hot(train_labels,num_labels)

#test baches 
test_images, test_labels = test_data['image'],test_data['label']
test_images = jnp.reshape(test_images, (len(test_images),num_pixels))
test_labels = one_hot(test_labels,num_labels)

os.system("clear")
print(train_images.shape, train_labels.shape)
print(test_images.shape,test_labels.shape)

params = init_network_params(layers_sizes,random.PRNGKey(0))

#training 
for epoch in range(num_epochs):
    start_time = time.time()
    for x,y in get_train_batches():
        x = jnp.reshape(x,(len(x),num_pixels))
        y = one_hot(y,num_labels)
        params = update(params, x, y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params,train_images,train_labels)
    test_acc = accuracy(params,test_images,test_labels)

    print(f"epoch {epoch} in {epoch_time}\ntrain_acc {train_acc}\ntest_acc {test_acc} \n")

