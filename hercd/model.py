from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module
import torch.nn.functional as F

from .constants import TERM_SIZE_LIMIT, WORD_EMBEDDING, HIDDEN_LAYER, DROPOUT

class Input(NamedTuple):
    """input for the model"""

    formula_words: Tensor
    """formula word tensor"""
    formula_positions: Tensor
    """formula position tensor"""
    target_words: Tensor
    """target word tensor"""
    target_positions: Tensor
    """target position tensor"""


class FormulaEmbedding(Module):
    """embed a formula"""

    embedding: Embedding
    """embed words"""
    mask: Tensor
    """a constant: all powers of 2 up to 2^WORD_EMBEDDING"""

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(TERM_SIZE_LIMIT, WORD_EMBEDDING, padding_idx=0)
        self.register_buffer('mask', 2 ** torch.arange(WORD_EMBEDDING), persistent=False)

    def forward(self, words: Tensor, positions: Tensor) -> Tensor:
        words = self.embedding(words)
        # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
        positions = positions.unsqueeze(-1).bitwise_and(self.mask).ne(0).long()
        x = words + positions
        return x.sum(dim=1)


class Model(Module):
    """the main model"""

    formula_embedding: FormulaEmbedding
    """embedding for the formula"""
    target_embedding: FormulaEmbedding
    """embedding for the target"""
    hidden: Linear
    """hidden layer"""
    dropout: Dropout
    """dropout"""
    output: Linear
    """output layer"""

    def __init__(self):
        super().__init__()
        self.formula_embedding = FormulaEmbedding()
        self.target_embedding = FormulaEmbedding()
        self.hidden = Linear(2 * WORD_EMBEDDING, HIDDEN_LAYER)
        self.dropout = Dropout(DROPOUT)
        self.output = Linear(HIDDEN_LAYER, 1)

    def forward(self, input: Input) -> Tensor:
        formula = self.formula_embedding(input.formula_words, input.formula_positions)
        target = self.target_embedding(input.target_words, input.target_positions)
        x = torch.cat((formula, target), dim=1)
        x = self.hidden(x)
        x = self.dropout(x)
        x = F.relu(x)
        return self.output(x).view(-1)
