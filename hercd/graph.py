from enum import IntEnum
import json

import torch
from torch_geometric.data import Data

from .cd import F

class Node(IntEnum):
    """a node in a `Graph`"""

    VAR = 0
    FUN = 1
    C1 = 2
    C2 = 3
    ENTRY = 4
    GOAL = 5

class Graph:
    """a graph object for representing CD states"""

    nodes: list[Node]
    """nodes in the graph"""
    sources: list[int]
    """indices of source nodes"""
    targets: list[int]
    """indices of target nodes"""
    entries: list[int]
    """indices of entry nodes"""
    pairs: list[tuple[int, int]]
    """entries which should be selected"""
    cache: dict[F, int]
    """map from subformulas to existing nodes"""

    def __init__(self, goal: F):
        self.nodes = []
        self.sources = []
        self.targets = []
        self.entries = []
        self.pairs = []
        self.cache = {}

        self.goal(goal)

    def entry(self, entry: F):
        """add an entry's formula to the graph"""
        formula = self._formula(entry)
        node = len(self.nodes)
        self.nodes.append(Node.ENTRY)
        self.sources.append(node)
        self.targets.append(formula)
        self.entries.append(node)

    def goal(self, goal: F):
        """add a goal formula to the graph"""
        formula = self._formula(goal)
        node = len(self.nodes)
        self.nodes.append(Node.GOAL)
        self.sources.append(node)
        self.targets.append(formula)

    def json(self) -> str:
        """produce a JSON representation of this graph"""
        return json.dumps({
            'nodes': self.nodes,
            'sources': self.sources,
            'targets': self.targets,
            'entries': self.entries,
            'pairs': self.pairs
        })

    def torch(self) -> Data:
        """produce a Torch representation of this graph"""
        return Data(
            x = torch.tensor(self.nodes),
            edge_index = torch.tensor([
                self.sources + self.targets,
                self.targets + self.sources
            ]),
            entry_index = torch.tensor(self.entries),
            pairs = torch.tensor(self.pairs)
        )

    def _formula(self, f: F) -> int:
        """add a formula to the graph"""
        if f in self.cache:
            return self.cache[f]

        if isinstance(f, int):
            node = len(self.nodes)
            self.nodes.append(Node.VAR)
        elif isinstance(f, str):
            node = len(self.nodes)
            self.nodes.append(Node.FUN)
        else:
            left = self._formula(f.left)
            right = self._formula(f.right)
            node = len(self.nodes)
            self.nodes.append(Node.C1)
            self.nodes.append(Node.C2)
            self.sources.append(node)
            self.targets.append(node + 1)
            self.sources.append(node + 1)
            self.targets.append(left)
            self.sources.append(node)
            self.targets.append(right)

        self.cache[f] = node
        return node
