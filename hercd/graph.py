from enum import IntEnum
from typing import Dict, List

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
    _cache: Dict[F, int]

    def __init__(self, sample: F, graph: F):
        self.nodes = [Node.SAMPLE, Node.GOAL]
        self.sources = [0]
        self.targets = [1]
        self._cache = {}

        node = self._formula(sample)
        self.sources.append(0)
        self.targets.append(node)
        self._cache.clear()

        node = self._formula(graph)
        self.sources.append(1)
        self.targets.append(node)
        self._cache.clear()

    def _formula(self, f: F) -> int:
        node = len(self.nodes)
        if f in self._cache:
            return self._cache[f]

        if isinstance(f, int):
            self.nodes.append(Node.VAR)
        elif isinstance(f, str):
            self.nodes.append(Node.FUN)
        else:
            self.nodes.append(Node.C1)
            self.nodes.append(Node.C2)
            left = self._formula(f.left)
            right = self._formula(f.right)
            self.sources.append(node)
            self.targets.append(node + 1)
            self.sources.append(node + 1)
            self.targets.append(left)
            self.sources.append(node)
            self.targets.append(right)

        self._cache[f] = node
        return node
