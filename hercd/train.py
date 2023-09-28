import gzip
import json

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch_geometric.data import Data, InMemoryDataset

from .model import Model

BATCH_SIZE = 64

class FileDataset(InMemoryDataset):
    """a dataset based on gzipped JSON lines representing a `Graph`"""

    file: str
    """path to gzip archive supplied by the user"""

    def __init__(self, file: str):
        self.file = file
        super().__init__('.')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return self.file

    @property
    def processed_file_names(self):
        return self.file.replace('.jsonl.gz', '.pt')

    def process(self):
        data = []
        with gzip.open(self.file, 'r') as stream:
            for line in stream:
                raw = json.loads(line)
                data.append(Data(
                    x = torch.tensor(raw['nodes']),
                    edge_index = torch.tensor([
                        raw['sources'] + raw['targets'],
                        raw['targets'] + raw['sources']
                    ]),
                    entry_index = torch.tensor(raw['entries']),
                    pairs = torch.tensor(raw['pairs'])
                ))
        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])

def compute_loss(model: Model, graph: Data) -> Tensor:
    """compute the loss for this graph"""

    # number of entries represented in the graph
    entries = graph.entry_index.shape[0]

    # matrix with zeroes everywhere except for coordinates in `graph.pairs`
    y = torch.zeros(entries * entries)
    indices = graph.pairs[:, 0] * entries + graph.pairs[:, 1]
    y[indices] = 1.0
    y = y.to('cuda').view(entries, entries)

    graph = graph.to('cuda')
    prediction = model(graph)
    return F.binary_cross_entropy_with_logits(prediction, y)

def train_from_file(path: str):
    """train a model from data provided in `path`"""
    dataset = FileDataset(path)
    model = Model().to('cuda')
    optimizer = Adam(model.parameters())

    step = 1
    writer = SummaryWriter()
    while True:
        for graph in dataset:
            assert isinstance(graph, Data)
            loss = compute_loss(model, graph)
            loss.backward()
            writer.add_scalar('loss', loss.detach(), global_step=step)
            if step % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            step += 1
