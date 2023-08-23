from torch.nn import functional as F
from torch.nn import Embedding, Module, ModuleList, Linear
from torch_geometric.nn import BatchNorm, GCNConv, global_add_pool

NODE_TYPES = 6
CHANNELS = 8
CONVOLUTIONS = 8
HIDDEN = 128


class Model(Module):
    embedding: Embedding
    bn: ModuleList
    conv: ModuleList
    hidden: Linear
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
        self.output = Linear(HIDDEN, 1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch = batch.batch
        x = self.embedding(x)
        for bn, conv in zip(self.bn, self.conv):
            convolved = conv(bn(x), edge_index)
            x = F.relu(x + convolved)
        x = global_add_pool(x, batch)
        x = F.relu(self.hidden(x))
        return self.output(x).squeeze(-1)
