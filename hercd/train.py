import gzip
import sys
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Optimizer, Adam

from .cd import F, C, N, from_repr, size
from .constants import BATCH_SIZE, TERM_SIZE_LIMIT
from .model import Input, Model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def move(x):
    """move a tensor or named tuple to `DEVICE`"""
    if isinstance(x, Tensor):
        return x.to(DEVICE)
    return x._make(move(field) for field in x)


class Word:
    N = 1
    C = 2

def _fill_formula_code(
    words: Tensor,
    positions: Tensor,
    f: F,
    offset: Tensor = torch.tensor(0),
    position: Tensor = torch.tensor(0)
):
    """fill `words` and `positions` with integers representing `f`"""

    assert offset < TERM_SIZE_LIMIT
    positions[offset] = int(position)
    if isinstance(f, int):
        words[offset] = Word.C + f
    elif isinstance(f, N):
        words[offset] = Word.N
        _fill_formula_code(words, positions, f.negated, offset + 1, (position << 2) | 1)
    elif isinstance(f, C):
        words[offset] = Word.C
        _fill_formula_code(words, positions, f.left, offset + 1, (position << 2) | 1)
        _fill_formula_code(words, positions, f.right, offset + size(f.left) + 1, (position << 2) | 2)


def encode_formula(f: F) -> tuple[Tensor, Tensor]:
    """encode a formula as word and position tensors"""
    words = torch.zeros(TERM_SIZE_LIMIT, dtype=torch.long)
    positions = torch.zeros(TERM_SIZE_LIMIT, dtype=torch.long)
    _fill_formula_code(words, positions, f)
    return words, positions


class Train(NamedTuple):
    """training example for the neural network"""

    input: Input
    """input tensors"""
    y: Tensor
    """output label"""


class Experience(NamedTuple):
    """a single member of the experience buffer"""

    formula: F
    """a sampled consequence"""
    target: F
    """a sampled hindsight goal"""
    train: Train
    """training data"""


class CDDataset(Dataset):
    """a trivial experienceset of formula/target/y triples"""

    experience: list[Experience]
    """list of triples"""

    def __init__(self, experience: list[Experience]):
        self.experience = experience

    @staticmethod
    def from_file(file: str) -> 'CDDataset':
        experience = []
        count = 0
        with gzip.open(file, 'rt') as f:
            for line in f:
                count += 1
                if count % 1000 == 0:
                    print('*', end='')
                    sys.stdout.flush()

                formula_repr, target_repr, y = line.split()
                formula = from_repr(formula_repr)
                target = from_repr(target_repr)
                y = torch.tensor(float(y))
                input = Input(
                    *encode_formula(formula),
                    *encode_formula(target)
                )
                train = Train(input, y)
                experience.append(Experience(formula, target, train))
        print()

        return CDDataset(experience)

    def __len__(self):
        return len(self.experience)

    def __getitem__(self, idx: int) -> Train:
        datum = self.experience[idx]
        return datum.train


def predict(model: Model, formulae: list[F], target: F) -> Tensor:
    """predict utility of `formulae` for `target`"""

    model.eval()
    target_tensors = encode_formula(target)
    batch = [
        Input(*encode_formula(formula), *target_tensors)
        for formula in formulae
    ]
    with torch.no_grad():
        return model(move(default_collate(batch)))


def forward(model: Model, train: Train) -> tuple[Tensor, Tensor]:
    """compute the loss for this batch"""
    logit = model(train.input)
    loss = binary_cross_entropy_with_logits(logit, train.y)
    return logit, loss


def create_optimizer(model: Model) -> Optimizer:
    """make an optimiser for `model`"""
    return Adam(model.parameters())


def validate(model: Model, dataset: Dataset, writer: SummaryWriter, step: int):
    """evaluate `model` on `dataset`"""

    model.eval()
    truth = []
    logits = []
    losses = []
    for batch in DataLoader(
        dataset,
        batch_size=BATCH_SIZE
    ):
        truth.append(batch.y)
        with torch.no_grad():
            batch = move(batch)
            assert isinstance(batch, Train)
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
        batch_size=BATCH_SIZE,
        shuffle=True
    ):
        batch = move(batch)
        assert isinstance(batch, Train)
        _, loss = forward(model, batch)
        loss.backward()
        if step % 10 == 0:
            writer.add_scalar('train/loss', loss.detach(), global_step=step)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    return step
