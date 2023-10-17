from enum import IntEnum

import torch
from torch_geometric.data import Data

from .cd import F, N

class Node(IntEnum):
    """a node in a `Graph`"""

    VAR = 0
    N = 1
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

    def __init__(self, entry: F, goal: F):
        self.nodes = []
        self.sources = []
        self.targets = []

        cache = {}
        self._marked(Node.ENTRY, cache, entry)
        self._marked(Node.GOAL, cache, goal)

    def torch(self) -> Data:
        """produce a Torch representation of this graph"""
        return Data(
            x = torch.tensor(self.nodes),
            edge_index = torch.tensor([
                self.sources + self.targets,
                self.targets + self.sources
            ])
        )

    def _marked(self, label: Node, cache: dict[F, int], entry: F):
        """add a marked formula to the graph"""
        formula = self._formula(cache, entry)
        node = len(self.nodes)
        self.nodes.append(label)
        self.sources.append(node)
        self.targets.append(formula)

    def _formula(self, cache: dict[F, int], f: F) -> int:
        """add a formula to the graph"""
        if f in cache:
            return cache[f]

        if isinstance(f, int):
            node = len(self.nodes)
            self.nodes.append(Node.VAR)
        elif isinstance(f, N):
            negated = self._formula(cache, f.negated)
            node = len(self.nodes)
            self.nodes.append(Node.N)
            self.targets.append(negated)
            self.sources.append(node)
        else:
            left = self._formula(cache, f.left)
            right = self._formula(cache, f.right)
            node = len(self.nodes)
            self.nodes.append(Node.C1)
            self.nodes.append(Node.C2)
            self.sources.append(node)
            self.targets.append(node + 1)
            self.sources.append(node + 1)
            self.targets.append(left)
            self.sources.append(node)
            self.targets.append(right)

        cache[f] = node
        return node
