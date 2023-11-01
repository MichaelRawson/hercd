import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Embedding, Linear, Module, ModuleList
from torch_geometric.data import Batch, Data
from torch_geometric.nn import InstanceNorm, MessagePassing, global_max_pool

from .graph import Graph

NODE_TYPES = 6
CHANNELS = 64
GIN_HIDDEN = 256
CONVOLUTIONS = 8
HIDDEN = 1024

class Summation(MessagePassing):
    """just sum neighbours"""

    def __init__(self, out: bool):
        flow = 'source_to_target' if out else 'target_to_source'
        super().__init__(flow=flow)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

class DirectedGINConv(Module):
    """directed version of the Graph Isomorphism Network"""

    out: Summation
    """outwards summation before pass through MLP"""
    back: Summation
    """backwards summation before pass through MLP"""
    hidden: Linear
    """hiden layer"""
    output: Linear
    """output layer"""

    def __init__(self):
        super().__init__()
        self.out = Summation(True)
        self.back = Summation(False)
        self.hidden = Linear(2 * CHANNELS, GIN_HIDDEN)
        self.output = Linear(GIN_HIDDEN, CHANNELS)

    def forward(self, x, edge_index) -> Tensor:
        out = self.out(x, edge_index)
        back = self.back(x, edge_index)
        x = torch.cat((out, back), dim=-1)
        x = self.hidden(x)
        x = F.relu_(x)
        return self.output(x)

class Model(Module):
    """classification of graphs"""

    embedding: Embedding
    """node embedding"""
    bn: ModuleList
    """batch normalisation layers"""
    conv: ModuleList
    """graph convolutional layers"""
    hidden: Linear
    """hidden layer"""
    output: Linear

    cache: dict[Graph, float]

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv = ModuleList([
            DirectedGINConv()
            for _ in range(CONVOLUTIONS)
        ])
        self.norm = ModuleList([
            InstanceNorm(CHANNELS)
            for _ in range(CONVOLUTIONS)
        ])
        self.hidden = Linear((CONVOLUTIONS + 1) * CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, 1)
        self.cache = {}

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
        return self.output(x).view(-1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cache.clear()

    def predict(self, graph: Graph) -> float:
        if graph in self.cache:
            return self.cache[graph]

        batch = Batch.from_data_list([graph.torch()]).to('cuda')
        prediction = float(torch.sigmoid(self(batch)))
        self.cache[graph] = prediction
        return prediction
