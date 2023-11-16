import random
from typing import Optional

import torch
from torch.distributions.categorical import Categorical

from .cd import C, Entry, F, TooBig, NoUnifier, match, modus_ponens
from .constants import FACT_LIMIT
from .model import Model

class Environment:
    """an environment for CD proofs"""

    axioms: list[Entry]
    """problem axioms"""
    goal: F
    """problem goal"""
    model: Optional[Model]
    """embedding network for policy - if None, apply a uniform policy"""
    chatty: bool
    """whether to output progress or not"""
    active: list[Entry]
    """the active set"""
    passive: list[Entry]
    """the passive set"""
    logits: list[float]
    """associated logits for the passive set from the model"""
    seen: set[F]
    """formulae we've already seen this episode"""

    def __init__(self, axioms: list[Entry], goal: F):
        self.axioms = axioms
        self.goal = goal
        self.model = None
        self.chatty = False
        self.reset()

    def reset(self):
        """reinitialise the environment, copying axioms to `known`"""
        self.active = []
        self.passive = []
        self.logits = []
        self.seen = set()
        self._add_to_passive(self.axioms)

    def run(self):
        """run an episode"""
        self.reset()
        if self.model:
            self.model.eval()

        while len(self.active) < FACT_LIMIT:
            self._activate(self._select())

    def sample(self) -> tuple[F, Entry, Entry]:
        """sample a hindsight goal and one positive/negative example"""

        while True:
            target = random.choice(self.passive)

            # positive examples are the ancestors of `target`
            positive = set(target.ancestors())

            # negatives are the rest
            negative = [
                entry
                for entry in self.active
                if entry.formula not in positive
            ]

            # selected an axiom or somehow all in `self.active` are ancestors, respectively
            if not positive or not negative:
                continue

            positive = random.choice(list(positive))
            negative = random.choice(negative)
            return target.formula, positive, negative

    def _select(self) -> Entry:
        """choose an entry from `self.passive`"""

        index = 0
        if self.model is None:
            index = random.randrange(len(self.passive))
        else:
            assert len(self.logits) == len(self.passive)
            logits = torch.tensor(self.logits)
            distribution = Categorical(logits=logits)
            index = distribution.sample()

        if self.model is not None:
            del self.logits[index]
        return self.passive.pop(index)

    def _activate(self, given: Entry):
        """activate `given`"""

        # re-check forward subsumption
        for other in self.active:
            if match(other.formula, given.formula):
                return

        # backward subsumption
        for index in reversed(range(len(self.active))):
            if match(given.formula, self.active[index].formula):
                del self.active[index]

        # we've committed to `given` now
        self.active.append(given)
        if self.chatty:
            print(len(self.active), given.formula)

        # do inference
        unprocessed = []
        for other in self.active:
            for major, minor in (other, given), (given, other):
                # see if they produce anything
                if not isinstance(major.formula, C):
                    continue
                try:
                    new = modus_ponens(major.formula.left, major.formula.right, minor.formula)
                except (NoUnifier, TooBig):
                    continue

                # skip duplicates
                if new in self.seen:
                    continue
                self.seen.add(new)

                # forwards subsumption
                if any(match(generalisation.formula, new) for generalisation in self.active):
                    continue

                unprocessed.append(Entry(new, major, minor))
        self._add_to_passive(unprocessed)

    def _add_to_passive(self, new: list[Entry]):
        """add `new` to `self.passive`"""

        if new and self.model is not None:
            self.logits.extend(self.model.predict(new, self.goal).tolist())
        self.passive.extend(new)
