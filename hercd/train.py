import gzip
import json
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam, Optimizer
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .model import Model

BATCH_SIZE = 64

class FileDataset(InMemoryDataset):
    file: str
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
                    y = torch.tensor(raw['y'], dtype=torch.float)
                ))
        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])

def train_step(model: Model, optimizer: Optimizer, batch) -> Tuple[Tensor, Tensor]:
    optimizer.zero_grad()
    batch = batch.to('cuda')
    prediction = model(batch)
    loss = F.binary_cross_entropy_with_logits(prediction, batch.y)
    loss.backward()
    optimizer.step()
    return prediction.detach(), loss.detach()

def train_from_file(path: str):
    dataset = FileDataset(path)
    model = Model().to('cuda')
    optimizer = Adam(model.parameters())

    step = 1
    writer = SummaryWriter()
    while True:
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
            prediction, loss = train_step(model, optimizer, batch)
            writer.add_scalar('loss', loss, global_step=step)
            if step % 100 == 0:
                print(torch.stack((torch.sigmoid(prediction).detach(), batch.y), dim=-1))
            step += 1
