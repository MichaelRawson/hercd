import random
from typing import Optional

from .cd import C, Entry, F, TooBig, NoUnifier, match, modus_ponens
from .constants import FACT_LIMIT
from .graph import Graph
from .model import Model

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
    cache: set[tuple[F, F]]
    """already tried an inference with these"""

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
        self.cache = set()
        for axiom in self.axioms:
            self._add(Entry(axiom))

    def run(self):
        """run an episode"""
        self.reset()

        if self.model:
            self.model.eval()

        while len(self.known) < FACT_LIMIT:
            new = self._step()
            self._add(new)
            if match(new.formula, self.goal):
                assert False, "proof found!"

    def sample(self) -> tuple[F, Entry, Entry]:
        """sample a hindsight goal and one positive/negative example"""

        while True:
            target = random.choice(self.known)

            # positive examples are the ancestors of `target`
            positive = {
                entry
                for entry in target.compacted_ancestors()
            }
            # negatives are the rest
            negative = {
                entry
                for entry in self.known
                if entry.formula not in positive
            }
            if not positive or not negative:
                continue

            positive = random.choice(list(positive))
            negative = random.choice(list(negative))
            return target.formula, positive, negative

    def _step(self) -> Entry:
        """deduce a new entry from `self.known`"""

        while True:
            # choose a random pair
            major, minor = random.choice(self.known), random.choice(self.known)
            if (major.formula, minor.formula) in self.cache:
                continue

            # see if they produce anything
            if not isinstance(major.formula, C):
                self.cache.add((major.formula, minor.formula))
                continue
            try:
                new = modus_ponens(major.formula.left, major.formula.right, minor.formula)
            except (NoUnifier, TooBig):
                self.cache.add((major.formula, minor.formula))
                continue

            # skip duplicates
            if new in self.seen:
                continue

            # forwards subsumption
            for other in self.known:
                if match(other.formula, new):
                    self.cache.add((major.formula, minor.formula))
                    continue

            entry = Entry(new, major, minor)
            # rejection sampling
            if self.model is not None:
                prediction = self.model.predict(entry, self.goal)
                threshold = random.random()
                if threshold >= prediction:
                    # don't add into cache here, might be selected later
                    continue

            return entry

    def _add(self, new: Entry):
        """add `new` to `self.known`, recording its parents"""
        self.seen.add(new.formula)

        # backwards subsumption
        for index in reversed(range(len(self.known))):
            if match(new.formula, self.known[index].formula):
                del self.known[index]

        if self.chatty:
            print(len(self.known), new.formula)
        self.known.append(new)
