import torch
from torch.nn import functional as F
from torch.nn import Embedding, Module, ModuleList, Linear
from torch_geometric.nn import BatchNorm, GCNConv

NODE_TYPES = 6
CHANNELS = 8
CONVOLUTIONS = 8
HIDDEN = 32

class Model(Module):
    """policy network"""

    embedding: Embedding
    """node embedding"""
    bn: ModuleList
    """batch normalisation layers"""
    conv: ModuleList
    """graph convolutional layers"""
    hidden: Linear
    """final hidden layer"""
    output: Linear
    """output layer"""

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
        self.hidden = Linear(2 * CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, 1)

    def forward(self, graph):
        entries = graph.entry_index.shape[0]
        line = torch.arange(entries).to('cuda')

        x = graph.x
        edge_index = graph.edge_index
        x = self.embedding(x)
        for bn, conv in zip(self.bn, self.conv):
            convolved = conv(bn(x), edge_index)
            x = F.relu(x + convolved)

        # project out the entries we care about
        x = x[graph.entry_index]

        # cartesian plane of all pairs of entry embeddings
        grid = torch.cartesian_prod(line, line).view(entries, entries, 2)
        x = x[grid].view(entries, entries, 2 * CHANNELS)
        x = F.relu(self.hidden(x))
        x = self.output(x).view(entries, entries)
        return x
