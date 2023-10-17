import gzip
import json
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Optimizer, Adam
from torch_geometric.data import Batch, Data

from .model import Model

ENTROPY_REGULARISATION = 0.7
BATCH_SIZE = 64

class CDDataset(Dataset):
    """a dataset based on gzipped JSON lines"""

    data: list[tuple[Data, Data, Tensor]]
    """major graph, minor graph, y"""

    def __init__(self, data: list[tuple[Data, Data, Tensor]]):
        self.data = data

    @staticmethod
    def from_file(file: str) -> 'CDDataset':
        data = []
        with gzip.open(file, 'r') as stream:
            for line in stream:
                raw = json.loads(line)
                data.append((
                    Data(
                        x = torch.tensor(raw['major']['nodes']),
                        edge_index = torch.tensor([
                            raw['major']['sources'] + raw['major']['targets'],
                            raw['major']['targets'] + raw['major']['sources']
                        ])
                    ),
                    Data(
                        x = torch.tensor(raw['minor']['nodes']),
                        edge_index = torch.tensor([
                            raw['minor']['sources'] + raw['minor']['targets'],
                            raw['minor']['targets'] + raw['minor']['sources']
                        ]),
                    ),
                    torch.tensor(float(raw['y']))
                ))
        return CDDataset(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    @staticmethod
    def collate(data: list[tuple[Data, Data, Tensor]]) -> tuple[Batch, Batch, Tensor]:
        major = Batch.from_data_list([major for major, _, _ in data])
        minor = Batch.from_data_list([minor for _, minor, _ in data])
        y = torch.tensor([y for _, _, y in data])
        assert isinstance(major, Batch) and isinstance(minor, Batch)
        return major, minor, y

def forward(model: Model, major: Data, minor: Data, y: Tensor) -> tuple[Tensor, Tensor]:
    """compute the loss for this batch"""
    major = major.to('cuda')
    minor = minor.to('cuda')
    y = y.to('cuda')
    major_embedding, minor_embedding = model(major, minor)
    logit = torch.sum(major_embedding * minor_embedding, dim=-1)
    xe = F.binary_cross_entropy_with_logits(logit, y)
    prediction = torch.sigmoid(logit)
    entropy = (torch.log(prediction) * prediction).mean()
    loss = xe + ENTROPY_REGULARISATION * entropy
    return prediction, loss

def epoch(
    model: Model,
    optimizer: Optimizer,
    dataset: CDDataset,
    writer: SummaryWriter,
    step: int,
    losses: Optional[list[float]] = None
) -> int:
    """train a `model` using `optimizer` from one pass through `dataset`"""
    for batch in DataLoader(dataset, collate_fn=CDDataset.collate, batch_size=BATCH_SIZE, shuffle=True):
            prediction, loss = forward(model, *batch)
            loss.backward()
            if losses is not None:
                losses.append(float(loss))
            writer.add_scalar('loss', loss.detach(), global_step=step)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 1000 == 0:
                writer.add_histogram('prediction', prediction, global_step=step)

    return step
