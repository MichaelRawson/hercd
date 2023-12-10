import gzip
import json

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Optimizer, AdamW
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from .constants import BATCH_SIZE
from .model import Model

class CDDataset(Dataset):
    """a dataset based on gzipped JSON lines"""

    data: list[BaseData]
    """list of graphs to be batched"""

    def __init__(self, data: list[BaseData]):
        self.data = data

    @staticmethod
    def from_file(file: str) -> 'CDDataset':
        data = []
        with gzip.open(file, 'r') as stream:
            for line in stream:
                raw = json.loads(line)
                data.append(Data(
                    x = torch.tensor(raw['nodes']),
                    edge_index = torch.tensor([raw['sources'], raw['targets']]),
                    y = torch.tensor(float(raw['y']))
                ))

        return CDDataset(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    @staticmethod
    def collate(data):
        batch = Batch.from_data_list(data)
        return batch

def forward(model: Model, batch: Data) -> tuple[Tensor, Tensor]:
    """compute the loss for this batch"""
    batch = batch.to('cuda')
    logit = model(batch)
    loss = binary_cross_entropy_with_logits(logit, batch.y)
    return logit, loss

def create_optimizer(model: Model) -> Optimizer:
    """make an optimiser for `model`"""
    return AdamW(model.parameters(), amsgrad=True, weight_decay=0.1)

def validate(model: Model, dataset: Dataset, writer: SummaryWriter, step: int):
    """evaluate `model` on `dataset`"""

    model.eval()
    truth = []
    logits = []
    losses = []
    for batch in DataLoader(
        dataset,
        collate_fn=CDDataset.collate,
        batch_size=BATCH_SIZE
    ):
        truth.append(batch.y)
        with torch.no_grad():
            logit, loss = forward(model, batch)
        logits.append(logit)
        losses.append(loss)

    truth = torch.cat(truth)
    logits = torch.cat(logits)
    loss = torch.tensor(losses).mean()
    writer.add_histogram('validation/distribution', logits, global_step=step)
    writer.add_scalar('validation/loss', loss, global_step=step)


def epoch(
    model: Model,
    optimizer: Optimizer,
    dataset: Dataset,
    writer: SummaryWriter,
    step: int
) -> int:
    """train a `model` for one epoch using `optimizer`"""

    model.train()
    for batch in DataLoader(
        dataset,
        collate_fn=CDDataset.collate,
        batch_size=BATCH_SIZE,
        shuffle=True
    ):
        _, loss = forward(model, batch)
        loss.backward()
        if step % 10 == 0:
            writer.add_scalar('train/loss', loss.detach(), global_step=step)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    return step
