import random
from typing import Generator, List, Optional, Set, Tuple

import torch

from .cd import C, F, TooBig, NoUnifier, match, modus_ponens
from .constants import STEP_LIMIT, EPSILON
from .graph import Graph
from .model import Model

class Entry:
    formula: F
    parents: Tuple['Entry']
    tree_size: int

    def __init__(self, term: F, parents: Tuple['Entry']):
        self.formula = term
        self.parents = parents
        self.tree_size = sum(parent.tree_size for parent in parents) + 1

    def ancestors(self) -> Generator['Entry', None, None]:
        for parent in self.parents:
            yield parent
            yield from parent.ancestors()

    def __repr__(self):
        return repr(self.formula)

class Environment:
    axioms: List[F]
    goal: F
    model: Optional[Model]
    known: List[Entry]
    discarded: Set[F]
    weights: List[float]
    seen: Set[F]
    proof: Optional[Entry]

    def __init__(self, axioms: List[F], goal: F, model: Optional[Model] = None):
        self.axioms = axioms
        self.goal = goal
        self.model = model
        self._reset()

    def run(self) -> bool:
        self._reset()

        while self.proof is None and len(self.known) < STEP_LIMIT:
            if random.random() < EPSILON:
                first, second = random.choices(self.known, k=2)
            else:
                weights = torch.softmax(torch.tensor(self.weights), 0).tolist()
                first, second = random.choices(self.known, weights=weights, k=2)

            for major, minor in (first.formula, second.formula), (second.formula, first.formula):
                if not isinstance(major, C):
                    continue
                try:
                    new = modus_ponens(major.left, major.right, minor)
                except (NoUnifier, TooBig):
                    continue
                self._add(new, first, second)

        return self.proof is not None

    def _reset(self):
        self.known = []
        self.discarded = set()
        self.weights = []
        self.seen = set()
        self.proof = None
        for axiom in self.axioms:
            self._add(axiom)

    def data(self) -> Tuple[F, Set[F], Set[F]]:
        target = self.proof if self.proof is not None else max(self.known, key=lambda entry: entry.tree_size)
        assert target is not None

        positive = {
            entry.formula
            for entry in iter(target.ancestors())
        }
        negative = self.discarded | {
            entry.formula
            for entry in self.known
            if entry.formula not in positive
        }

        return target.formula, negative, positive

    def _add(self, formula: F, *parents: Entry):
        if formula in self.seen:
            return
        self.seen.add(formula)

        for known in self.known:
            if match(known.formula, formula):
                self.discarded.add(formula)
                return

        # will be retained at this point
        print(len(self.known), formula)

        for index in reversed(range(len(self.known))):
            if match(formula, self.known[index].formula):
                self.discarded.add(self.known[index].formula)
                del self.known[index]
                del self.weights[index]

        entry = Entry(
            formula,
            parents,
        )
        if match(formula, self.goal):
            self.proof = entry
            print("proof!")
            return

        self.known.append(entry)
        self.weights.append(self._score(formula))
        return

    def _score(self, formula: F) -> float:
        if self.model is None:
            return 1.0

        datum = Graph(formula, self.goal).torch(False).to('cuda')
        self.model.eval()
        return self.model(datum).item()
