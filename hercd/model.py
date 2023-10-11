from torch import Tensor
from torch.nn import functional as F
from torch.nn import Embedding, Linear, Module, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool

NODE_TYPES = 6
CHANNELS = 64
CONVOLUTIONS = 8
HIDDEN = 128

class Head(Module):
    """embedding network"""

    embedding: Embedding
    """node embedding"""
    bn: ModuleList
    """batch normalisation layers"""
    conv: ModuleList
    """graph convolutional layers"""
    hidden: Linear
    """hidden layer"""
    output: Linear

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.bn = ModuleList([
            BatchNorm(CHANNELS)
            for _ in range(CONVOLUTIONS)
        ])
        self.conv = ModuleList([
            GCNConv(CHANNELS, CHANNELS)
            for _ in range(CONVOLUTIONS)
        ])
        self.hidden = Linear(CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, CHANNELS)

    def forward(self, graph: Data) -> Tensor:
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        x = self.embedding(x)
        for bn, conv in zip(self.bn, self.conv):
            convolved = conv(bn(x), edge_index)
            x = F.relu(x + convolved)
        x = global_mean_pool(x, batch)
        x = F.relu(self.hidden(x))
        return self.output(x)

class Model(Module):
    """embedding of major/minor pairs"""

    major: Module
    """major embedding"""
    minor: Module
    """minor embedding"""

    def __init__(self):
        super().__init__()
        self.major = Head()
        self.minor = Head()

    def forward(self, major: Data, minor: Data) -> tuple[Tensor, Tensor]:
        return self.major(major), self.minor(minor)
