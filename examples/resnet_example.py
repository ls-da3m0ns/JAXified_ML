import jax.numpy as jnp
from jax import jit, grad, random
import numpy.random as npr
from jax.experimental import optimizers
from resnet import *


bt_size = 4
img_size = (48,48)
channels = 3
total_imgs = 1024
inp_size = (bt_size,img_size[0],img_size[1],channels)



key = random.PRNGKey(0)
demo_imgs = random.randint(key,
                          shape=(total_imgs,img_size[0],img_size[0],channels,),
                           minval = 0,
                           maxval = 255
                          )
demo_labels = random.randint(key,
                            shape=(total_imgs,),
                            minval=0,
                            maxval=10)
print(demo_imgs.shape)
print(demo_labels.shape)



if __name__ == "__main__":
  rng_key = random.PRNGKey(0)

  batch_size = 4
  num_classes = 1001
  input_shape = (224, 224, 3, batch_size)
  step_size = 0.1
  num_steps = 10

  init_fun, predict_fun = ResNet(num_classes)
  _, init_params = init_fun(rng_key, input_shape)

  def loss(params, batch):
    inputs, targets = batch
    logits = predict_fun(params, inputs)
    return -jnp.sum(logits * targets)

  def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
    return jnp.mean(predicted_class == target_class)

  def synth_batches():
    rng = npr.RandomState(0)
    while True:
      images = rng.rand(*input_shape).astype('float32')
      labels = rng.randint(num_classes, size=(batch_size, 1))
      onehot_labels = labels == jnp.arange(num_classes)
      yield images, onehot_labels

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
  batches = synth_batches()

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  opt_state = opt_init(init_params)
  for i in range(num_steps):
    opt_state = update(i, opt_state, next(batches))
  trained_params = get_params(opt_state)
