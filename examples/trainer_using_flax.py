import torch.utils.data as torch_data
import flax 
from flax import linen
from jaxified import LeNet5
#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
class TestModel(linen.Module):
    n_classes: int
    def setup(self):
        self.le_net = LeNet5(self.n_classes)
    
    def __call__(self, x):
        return self.le_net(x)

TestModel(n_classes=10)
