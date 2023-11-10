from array import array

import torch
from torch_geometric.data import Data

from .cd import F, N, size, height
from .environment import Entry

class Node:
    VAR = 0
    N = 1
    C1 = 2
    C2 = 3
    END = 4

def feature(flavour: int, goal: bool, tree_size: int, height: int) -> array:
    """make a 1-dimensional feature vector for a node"""
    result = array('f')
    for i in range(Node.END):
        result.append(flavour == i)
    result.append(goal)
    result.append(tree_size)
    result.append(height)
    return result

class Graph:
    """a graph object for representing CD states"""

    nodes: list[array]
    """node feature vectors"""
    sources: array
    """indices of source nodes"""
    targets: array
    """indices of target nodes"""
    meta: array
    """metadata for the whole graph"""
    y: float
    """target value, if available"""
    entry: str
    """the entry this was constructed from"""
    goal: str
    """the goal this was constructed from"""

    def __init__(self, entry: Entry, goal: F):
        self.nodes = []
        self.sources = array('L')
        self.targets = array('L')
        self.meta = array('f')
        self.y = 0.0
        self.entry = str(entry.formula)
        self.goal = str(goal)

        self.meta.append(entry.tree_size())
        self.meta.append(entry.compacted_size())
        self.meta.append(entry.height())
        self.meta.append(size(entry.formula))
        self.meta.append(height(entry.formula))
        self.meta.append(size(goal))
        self.meta.append(height(goal))

        cache = {}
        self._formula(cache, goal, goal=True)
        self._formula(cache, entry.formula, goal=False)

    def torch(self) -> Data:
        """produce a Torch representation of this graph"""
        return Data(
            x = torch.tensor(self.nodes),
            edge_index = torch.tensor([self.sources, self.targets]),
            meta = torch.tensor([self.meta]),
            y = torch.tensor(self.y)
        )

    def _formula(self, cache: dict[F, int], f: F, goal: bool) -> int:
        """add a formula to the graph"""
        if f in cache:
            return cache[f]

        if isinstance(f, int):
            node = len(self.nodes)
            self.nodes.append(feature(Node.VAR, goal, 1, 1))
        elif isinstance(f, N):
            negated = self._formula(cache, f.negated, goal)
            node = len(self.nodes)
            self.nodes.append(feature(Node.N, goal, f.size, f.height))
            self.targets.append(negated)
            self.sources.append(node)
        else:
            left = self._formula(cache, f.left, goal)
            right = self._formula(cache, f.right, goal)
            node = len(self.nodes)
            self.nodes.append(feature(Node.C1, goal, f.size, f.height))
            self.nodes.append(feature(Node.C2, goal, f.size, f.height))
            self.sources.append(node)
            self.targets.append(node + 1)
            self.sources.append(node + 1)
            self.targets.append(left)
            self.sources.append(node)
            self.targets.append(right)

        cache[f] = node
        return node
