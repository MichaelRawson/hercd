import gzip
import json

import torch
from torch import Tensor
from torch.nn import functional as F
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
                    meta = torch.tensor([raw['meta']]),
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
    loss = F.binary_cross_entropy_with_logits(logit, batch.y)
    prediction = torch.sigmoid(logit)
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
        truth.append(batch.y)
        with torch.no_grad():
            prediction, loss = forward(model, batch)
        predictions.append(prediction)
        losses.append(loss)

    truth = torch.cat(truth)
    distribution = torch.cat(predictions)
    loss = torch.tensor(losses).mean()
    writer.add_histogram('validation/distribution', distribution, global_step=step)
    writer.add_pr_curve('validation/pr', truth, distribution, global_step=step)
    writer.add_scalar('validation/loss', loss, global_step=step)

def validate_steps(model: Model, dataset: Dataset, writer: SummaryWriter, step: int):
    """evaluate `model` on steps of a proof"""

    model.eval()
    predictions = []
    for batch in DataLoader(
        dataset,
        collate_fn=CDDataset.collate,
        batch_size=BATCH_SIZE
    ):
        with torch.no_grad():
            prediction, _ = forward(model, batch)
        predictions.append(prediction)

    distribution = torch.cat(predictions)
    writer.add_histogram('steps/distribution', distribution, global_step=step)


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
        shuffle=True
    ):
        _, loss = forward(model, batch)
        loss.backward()
        writer.add_scalar('train/loss', loss.detach(), global_step=step)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    return step
