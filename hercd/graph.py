from enum import IntEnum
from typing import Dict, List

import torch
from torch_geometric.data import Data

from .cd import F

"""a node in a graph"""
class Node(IntEnum):
    VAR = 0
    FUN = 1
    C1 = 2
    C2 = 3
    SAMPLE = 4
    GOAL = 5

"""list of nodes and edges"""
class Graph:
    """nodes in the graph"""
    nodes: List[Node]
    """indices of source nodes"""
    sources: List[int]
    """indices of target nodes"""
    targets: List[int]

    def __init__(self, sample: F, target: F):
        self.nodes = [Node.SAMPLE, Node.GOAL]
        self.sources = [0]
        self.targets = [1]
        cache = {}

        node = self._formula(cache, sample)
        self.sources.append(0)
        self.targets.append(node)
        cache.clear()

        node = self._formula(cache, target)
        self.sources.append(1)
        self.targets.append(node)

    def torch(self, y: bool):
        return Data(
            x = torch.tensor(self.nodes),
            edge_index = torch.tensor([
                self.sources + self.targets,
                self.targets + self.sources
            ]),
            y = torch.tensor(float(y))
        )

    def _formula(self, cache: Dict[F, int], f: F) -> int:
        node = len(self.nodes)
        if f in cache:
            return cache[f]

        if isinstance(f, int):
            self.nodes.append(Node.VAR)
        elif isinstance(f, str):
            self.nodes.append(Node.FUN)
        else:
            self.nodes.append(Node.C1)
            self.nodes.append(Node.C2)
            left = self._formula(cache, f.left)
            right = self._formula(cache, f.right)
            self.sources.append(node)
            self.targets.append(node + 1)
            self.sources.append(node + 1)
            self.targets.append(left)
            self.sources.append(node)
            self.targets.append(right)

        cache[f] = node
        return node
