from itertools import chain
import random
from typing import Generator, List, Optional, Set, Tuple

import torch

from .cd import C, F, TooBig, NoUnifier, match, modus_ponens
from .constants import ACTIVATION_LIMIT
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
    active: List[Entry]
    passive: List[Entry]
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
        for axiom in self.axioms:
            self._enqueue(axiom)

        step = 0
        while self.proof is None and self.passive and step < ACTIVATION_LIMIT:
            given = self._select()
            if given is None:
                continue
            print(given)
            self._activate(given)
            step += 1

        return self.proof is not None

    def _reset(self):
        self.active = []
        self.passive = []
        self.weights = []
        self.seen = set()
        self.proof = None

    def sample(self) -> Tuple[F, List[F], List[F]]:
        target = self.proof if self.proof is not None else self.active[-1]
        assert target is not None

        positive = {
            entry.formula
            for entry in iter(target.ancestors())
        }
        negative = {
            entry.formula
            for entry in self.active
            if entry.formula not in positive
        }

        sample_size = min(len(positive), len(negative))
        negative_sample = random.sample(tuple(negative), sample_size)
        positive_sample = random.sample(tuple(positive), sample_size)
        return target.formula, negative_sample, positive_sample

    def _select(self) -> Optional[Entry]:
        weights = torch.softmax(torch.tensor(self.weights), 0).tolist()
        index = random.choices(range(len(self.passive)), weights=weights)[0]
        given = self.passive.pop(index)
        self.weights.pop(index)
        if any(match(generalisation.formula, given.formula) for generalisation in self.active):
            return None
        return given

    def _activate(self, given: Entry):
        self.active.append(given)
        for partner in self.active:
            for major, minor in (partner.formula, given.formula), (given.formula, partner.formula):
                if not isinstance(major, C):
                    continue
                try:
                    new = modus_ponens(major.left, major.right, minor)
                except (NoUnifier, TooBig):
                    continue
                self._enqueue(new, given, partner)

    def _enqueue(self, formula: F, *parents: Entry):
        if formula in self.seen:
            return
        self.seen.add(formula)

        entry = Entry(
            formula,
            parents,
        )
        if match(formula, self.goal):
            self.proof = entry
            print("proof!")
            return

        self.passive.append(entry)
        self.weights.append(self._score(formula))

    def _score(self, formula: F) -> float:
        if self.model is None:
            return 1.0

        datum = Graph(formula, self.goal).torch(False).to('cuda')
        self.model.eval()
        return self.model(datum).item()
