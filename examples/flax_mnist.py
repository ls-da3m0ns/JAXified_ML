from typing import Any
from torchvision import datasets , transforms
import flax 
import jax
import jax.numpy as jnp
from flax import linen as nn
import torch 
from jax.typing import ArrayLike
import optax
from flax import struct 
from flax.training import train_state
from clu import metrics
from tqdm import tqdm

## Data loading
def load_mnist_data(batch_size: int, train_transforms = None, test_transforms= None, max_epochs: int = 10):
    """
    Load MNIST data from torchvision.datasets.MNIST
    uses dataloaders from torch to load data
    """
    train = datasets.MNIST('data', 
                train=True, 
                download=True,
                transform=train_transforms,
                )
    test = datasets.MNIST('data',
                train=False,
                download=True,
                transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader


## model definition


class CustomCNN( nn.Module ):
    """
        Basic Cnn based model to perform classification on MNIST data
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x
    
@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average = struct.field(pytree_node=False)
    accuracy: metrics.Accuracy.from_output('loss') = struct.field(pytree_node=False)


class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state( module , rng, learning_rate, momentum):
    params = module.init(rng, jnp.ones((1, 28, 28, 1)))[ 'params' ]
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics = Metrics.empty()
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn( {'params': params}, batch['image'] )
        loss = optax.softmax_cross_entropy( nn.one_hot(logits), nn.one_hot(batch['label']) ) 
        return jnp.mean(loss)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn( {'params': state.params}, batch['image'] )
    loss = optax.softmax_cross_entropy( nn.one_hot(logits), nn.one_hot(batch['label']) ) 
    accuracy = jnp.mean( jnp.argmax(logits, axis=-1) == batch['label'] )
    metrics_update = state.metrics.single_from_model_output( loss=loss,label=batch['label'], logits=logits )
    return state.replace( metrics = metrics_update )



## testing model 
if __name__ == "__main__":
    cnn = CustomCNN()
    print("Model Summary")
    print(cnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1))))

    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9

    train_loader, test_loader = load_mnist_data(
            batch_size=batch_size,
            train_transforms=transforms.Compose([
                transforms.ToTensor(),
            ]),
            test_transforms=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )
    init_rng = jax.random.PRNGKey(0)

    state = create_train_state(cnn, init_rng, learning_rate, momentum)
    del init_rng

    num_steps_per_epoch = len(train_loader)

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    for epoch in range(num_epochs):
        # Train
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}", total=num_steps_per_epoch):
            batch = {
                "image" : jnp.array(batch[0]),
                "label" : jnp.array(batch[1])
            }
            state = train_step(state, batch)

        # Evaluate
        for batch in tqdm(test_loader, desc=f"Evaluating epoch {epoch}", total=len(test_loader)):
            batch = {
                "image" : jnp.array(batch[0]),
                "label" : jnp.array(batch[1])
            }
            state = eval_step(state, batch)

        # Save metrics
        metrics_history['train_loss'].append(state.metrics['loss'])
        metrics_history['train_accuracy'].append(state.metrics['accuracy'])
        metrics_history['test_loss'].append(state.metrics['loss'])
        metrics_history['test_accuracy'].append(state.metrics['accuracy'])

        # Print metrics
        print(f"Epoch {epoch}: train_loss: {state.metrics['loss']}, train_accuracy: {state.metrics['accuracy']}, test_loss: {state.metrics['loss']}, test_accuracy: {state.metrics['accuracy']}")




    