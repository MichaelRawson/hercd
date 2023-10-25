import gzip
import json

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Optimizer, AdamW
from torch_geometric.data import Batch, Data

from .model import Model

BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

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
    prediction = torch.sigmoid(logit)
    loss = F.binary_cross_entropy_with_logits(logit, y)
    return prediction, loss

def create_optimizer(model: Model) -> Optimizer:
    """make an optimiser for `model`"""
    return AdamW(model.parameters(), amsgrad=True)

def validate(model: Model, dataset: Dataset, writer: SummaryWriter, step: int):
    """evaluate `model` on `dataset`"""

    model.eval()
    truth = []
    predictions = []
    losses = []
    for batch in DataLoader(
        dataset,
        collate_fn=CDDataset.collate,
        batch_size=BATCH_SIZE
    ):
        truth.append(batch[2])
        with torch.no_grad():
            prediction, loss = forward(model, *batch)
        predictions.append(prediction)
        losses.append(loss)

    truth = torch.cat(truth)
    prediction = torch.cat(predictions)
    loss = torch.tensor(losses).mean()
    writer.add_histogram('validation/prediction', prediction, global_step=step)
    writer.add_pr_curve('validation/pr', truth, prediction, global_step=step)
    writer.add_scalar('validation/loss', loss, global_step=step)


def epoch(
    model: Model,
    optimizer: Optimizer,
    dataset: Dataset,
    writer: SummaryWriter,
    step: int
) -> int:
    """train a `model` using `optimizer` from one pass through `dataset`"""

    model.train()
    for batch in DataLoader(
        dataset,
        collate_fn=CDDataset.collate,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    ):
        _, loss = forward(model, *batch)
        loss.backward()
        writer.add_scalar('train/loss', loss.detach(), global_step=step)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    return step
