from array import array

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

from .cd import F, N
from .environment import Entry

class Node:
    VAR = 0
    N = 1
    C1 = 2
    C2 = 3
    ENTRY = 4
    GOAL = 5
    D = 6
    END = 7

CACHE: dict[F, int] = {}
"""cache nodes representing formulae"""

class Graph:
    """a graph object for representing CD states"""

    nodes: array
    """node feature vectors"""
    sources: array
    """indices of source nodes"""
    targets: array
    """indices of target nodes"""
    y: float
    """target value, if available"""
    entry: str
    """the entry this was constructed from"""
    goal: str
    """the goal this was constructed from"""

    def __init__(self, entry: Entry, goal: F):
        self.nodes = array('L')
        self.sources = array('L')
        self.targets = array('L')
        self.y = 0.0
        self.entry = str(entry.formula)
        self.goal = str(goal)

        CACHE.clear()
        self._node(Node.ENTRY, self._formula(entry.formula))
        self._node(Node.GOAL, self._formula(goal))

    def torch(self) -> BaseData:
        """produce a Torch representation of this graph"""
        return Data(
            x = torch.tensor(self.nodes),
            edge_index = torch.tensor([self.sources, self.targets]),
            y = torch.tensor(self.y)
        )

    def _formula(self, f: F) -> int:
        """add a formula to the graph"""

        # cache previously-seen formulae
        if f in CACHE:
            return CACHE[f]

        if isinstance(f, int):
            node = self._node(Node.VAR)
        elif isinstance(f, N):
            negated = self._formula(f.negated)
            node = self._node(Node.N, negated)
        else:
            left = self._formula(f.left)
            right = self._formula(f.right)
            left = self._node(Node.C2, left)
            node = self._node(Node.C1, left, right)

        CACHE[f] = node
        return node

    def _node(self, label: int, *children: int) -> int:
        """add a node with `label` and `children`"""

        index = len(self.nodes)
        self.nodes.append(label)
        for child in children:
            self.sources.append(index)
            self.targets.append(child)
        return index
