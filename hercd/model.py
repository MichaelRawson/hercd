import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Embedding, Linear, Module, ModuleList
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import InstanceNorm, GINConv, global_max_pool

NODE_TYPES = 6
CHANNELS = 32
GIN_HIDDEN = 64
CONVOLUTIONS = 8
HIDDEN = 64
OUTPUT = 32

class GINApproximator(Module):
    """MLP for the GIN"""

    hidden: Linear
    """hiden layer"""
    output: Linear
    """output layer"""

    def __init__(self):
        super().__init__()
        self.hidden = Linear(CHANNELS, GIN_HIDDEN)
        self.output = Linear(GIN_HIDDEN, CHANNELS)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        x = F.relu_(x)
        return self.output(x)

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
        self.conv = ModuleList([
            GINConv(GINApproximator())
            for _ in range(CONVOLUTIONS)
        ])
        self.norm = ModuleList([
            InstanceNorm(CHANNELS)
            for _ in range(CONVOLUTIONS)
        ])
        self.hidden = Linear((CONVOLUTIONS + 1) * CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, OUTPUT)

    def forward(self, graph: Data) -> Tensor:
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        x = self.embedding(x)
        xs = [x]
        for conv, norm in zip(self.conv, self.norm):
            x = F.relu_(x + norm(conv(x, edge_index)))
            xs.append(x)
        xs = torch.cat(xs, dim=1)
        x = global_max_pool(xs, batch)
        x = F.relu_(self.hidden(x))
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
