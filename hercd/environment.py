from collections.abc import Generator
import random
from typing import Optional

from .cd import C, F, TooBig, NoUnifier, match, modus_ponens
from .constants import FACT_LIMIT
from .graph import Graph
from .model import Model

class Entry:
    """a deduced formula and its proof"""

    formula: F
    """the formula that this entry proves"""
    parents: tuple['Entry', ...]
    """premises - empty for axioms"""
    tree_size: int
    """the tree size of the deduction"""

    def __init__(self, term: F, *parents: 'Entry'):
        self.formula = term
        self.parents = parents
        self.tree_size = sum(parent.tree_size for parent in parents) + 1

    def ancestors(self) -> Generator['Entry', None, None]:
        """this entry's parents and all their parents, recursively"""
        for parent in self.parents:
            yield parent
            yield from parent.ancestors()

class Environment:
    """an environment for CD proofs"""

    axioms: list[F]
    """problem axioms"""
    goal: F
    """problem goal"""
    model: Optional[Model]
    """embedding network for policy - if None, apply a uniform policy"""
    chatty: bool
    """whether to be chatty or not"""
    known: list[Entry]
    """deduced so far"""
    seen: set[F]
    """formulae we've already seen this episode"""
    proof: Optional[Entry]
    """if not None, a proof of `goal` from `axioms`"""

    def __init__(self, axioms: list[F], goal: F):
        self.axioms = axioms
        self.goal = goal
        self.model = None
        self.chatty = False
        self.reset()

    def reset(self):
        """reinitialise the environment, copying axioms to `known`"""
        self.known = []
        self.seen = set()
        self.proof = None
        for axiom in self.axioms:
            self._add(Entry(axiom))

    def run(self) -> bool:
        """run an episode, returning True if we found a proof"""
        self.reset()

        if self.model:
            self.model.eval()

        while self.proof is None and len(self.known) < FACT_LIMIT:
            new = self._step()
            self._add(new)

        return self.proof is not None

    def training(self) -> tuple[F, set[F], set[F]]:
        """produce a hindsight goal and positive/negative examples"""

        # find what we're aiming for
        target = self.proof if self.proof is not None else max(self.known, key=lambda entry: entry.tree_size)
        assert target is not None

        # positive examples are the ancestors of `target`
        positive = {
            entry.formula
            for entry in target.ancestors()
        }
        # negatives are the rest
        negative = {
            entry.formula
            for entry in self.known
            if entry.formula not in positive
        }

        return target.formula, positive, negative

    def _step(self) -> Entry:
        """deduce a new entry from `self.known`"""

        while True:
            # choose a random pair
            major, minor = random.choice(self.known), random.choice(self.known)

            # see if they produce anything
            if not isinstance(major.formula, C):
                continue
            try:
                new = modus_ponens(major.formula.left, major.formula.right, minor.formula)
            except (NoUnifier, TooBig):
                continue

            # forwards subsumption
            for other in self.known:
                if match(other.formula, new):
                    continue

            # rejection sampling
            if self.model is not None:
                graph = Graph(new, self.goal)
                prediction = self.model.predict(graph)
                threshold = random.random()
                if threshold >= prediction:
                    continue

            return Entry(new, major, minor)

    def _add(self, new: Entry):
        """add `new` to `self.known`, recording its parents"""

        # skip duplicates
        if new.formula in self.seen:
            return
        self.seen.add(new.formula)

        # backwards subsumption
        for index in reversed(range(len(self.known))):
            if match(new.formula, self.known[index].formula):
                del self.known[index]

        if self.chatty:
            print(len(self.known), new.formula)

        if match(new.formula, self.goal):
            self.proof = new
            if self.chatty:
                print("proof!")
            return

        self.known.append(new)
