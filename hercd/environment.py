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
    parents: tuple['Entry', 'Entry']
    """premises - empty for axioms"""
    tree_size: int
    """the tree size of the deduction"""

    def __init__(self, term: F, parents: tuple['Entry', 'Entry']):
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
    active: list[Entry]
    """the active set"""
    passive: list[Entry]
    """the passive set"""
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
        self.active = []
        self.passive = []
        self.seen = set()
        self.log = []
        self.proof = None
        for axiom in self.axioms:
            self._add(axiom)

    def run(self) -> bool:
        """run an episode, returning True if we found a proof"""
        self.reset()

        if self.model:
            self.model.eval()

        while self.proof is None and len(self.active) < FACT_LIMIT:
            selected_index = self._select()
            selected = self.passive.pop(selected_index)
            self._activate(selected)

        return self.proof is not None

    def training(self) -> tuple[F, set[F], set[F]]:
        """produce a hindsight goal and positive/negative examples"""

        # find what we're aiming for
        target = self.proof if self.proof is not None else max(self.passive, key=lambda entry: entry.tree_size)
        assert target is not None

        # positive examples are the ancestors of `target`
        positive = {
            entry.formula
            for entry in target.ancestors()
        }
        # negatives are the rest
        negative = {
            entry.formula
            for entry in self.active
            if entry.formula not in positive
        }

        return target.formula, positive, negative

    def _select(self) -> int:
        """choose a passive clause and return its index"""
        if self.model is None:
            return random.randrange(len(self.passive))

        # rejection sampling
        while True:
            candidate = random.randrange(len(self.passive))
            graph = Graph(self.passive[candidate].formula, self.goal)
            prediction = self.model.predict(graph)
            threshold = random.random()
            if threshold < prediction:
                return candidate

    def _activate(self, selected: Entry):
        """activate `selected`"""

        # forwards subsumption
        for index in reversed(range(len(self.passive))):
            if match(selected.formula, self.passive[index].formula):
                del self.passive[index]

        # subsume things in the active set
        for index in reversed(range(len(self.active))):
            if match(selected.formula, self.active[index].formula):
                del self.active[index]

        self.active.append(selected)
        for other in self.active:
            for major, minor in (selected.formula, other.formula), (other.formula, selected.formula):
                if not isinstance(major, C):
                    continue
                try:
                    new = modus_ponens(major.left, major.right, minor)
                except (NoUnifier, TooBig):
                    continue

                self._add(new, other, selected)

        if self.chatty:
            print(len(self.active), selected.formula)

    def _add(self, new: F, *parents: Entry):
        """add `new` to `passive`, recording its parents"""

        # skip duplicates
        if new in self.seen:
            return
        self.seen.add(new)

        # forwards subsumption
        for other in self.active:
            if match(other.formula, new):
                return

        entry = Entry(new, parents)
        if match(new, self.goal):
            self.proof = entry
            if self.chatty:
                print("proof!")
            return

        self.passive.append(entry)
