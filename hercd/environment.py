from heapq import heappush, heappop
from itertools import chain
import random
from typing import Generator, List, Optional, Tuple

from .cd import C, F, TooBig, NoUnifier, match, modus_ponens
from .constants import ACTIVATION_LIMIT

class Entry:
    formula: F
    parents: Tuple['Entry']
    priority: float
    tree_size: int

    def __init__(self, term: F, parents: Tuple['Entry']):
        self.formula = term
        self.parents = parents
        self.priority = 0
        self.tree_size = sum(parent.tree_size for parent in parents) + 1

    def ancestors(self) -> Generator['Entry', None, None]:
        for parent in self.parents:
            yield parent
            yield from parent.ancestors()

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return repr(self.formula)

class Environment:
    axioms: List[F]
    goal: F
    active: List[Entry]
    passive: List[Entry]
    proof: Optional[Entry]
    largest: Optional[Entry]

    def __init__(self, axioms: List[F], goal: F):
        self.axioms = axioms
        self.goal = goal
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
            self._activate(given)
            step += 1

        return self.proof is not None

    def _reset(self):
        self.active = []
        self.passive = []
        self.proof = None
        self.largest = None

    def sample(self) -> Tuple[F, List[F], List[F]]:
        target = self.proof if self.proof is not None else self.largest
        assert target is not None

        positive = {
            entry.formula
            for entry in iter(target.ancestors())
        }
        negative = {
            entry.formula
            for entry in chain(self.active, self.passive)
            if entry.formula not in positive
        }

        sample_size = min(len(positive), len(negative))
        negative_sample = random.sample(tuple(negative), sample_size)
        positive_sample = random.sample(tuple(positive), sample_size)
        return target.formula, negative_sample, positive_sample

    def _select(self) -> Optional[Entry]:
        given = heappop(self.passive)
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
        entry = Entry(
            formula,
            parents,
        )
        if match(formula, self.goal):
            self.proof = entry
            # print("proof!")
            return

        if self.largest is None or entry.tree_size >= self.largest.tree_size:
            self.largest = entry
        entry.priority = self._score(formula)
        heappush(self.passive, entry)

    def _score(self, formula: F) -> float:
        return random.random()
