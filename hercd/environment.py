from array import array
import random
from typing import Optional

import torch
from torch.distributions import Categorical

from .cd import C, Entry, F, TooBig, NoUnifier, match, modus_ponens, n_simplify
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
    logits: array
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
        self.logits = array('f')
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

        if self.model is None:
            index = random.randrange(len(self.passive))
        else:
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
        deleted = set()
        for index in reversed(range(len(self.active))):
            if match(given.formula, self.active[index].formula):
                deleted.add(self.active.pop(index))

        if deleted:
            # orphaned active clauses
            index = len(self.active)
            while index > 0:
                index -= 1
                if any(parent in deleted for parent in self.active[index].parents):
                    deleted.add(self.active.pop(index))
                    index = len(self.active)

            # orphaned passive clauses
            for index in reversed(range(len(self.passive))):
                if any(parent in deleted for parent in self.passive[index].parents):
                    if self.model:
                        del self.logits[index]
                    del self.passive[index]

        # we've committed to `given` now
        self.active.append(given)
        if self.chatty:
            print(len(self.active), given.formula)

        # do inference
        unprocessed = []
        if given.n_simplify:
            new = n_simplify(given)
            if self._retain(new):
                unprocessed.append(Entry(new, given))

        for other in self.active:
            for major, minor in (other, given), (given, other):
                # see if they produce anything
                if not isinstance(major.formula, C) or major.n_simplify:
                    continue

                left = major.formula.left
                right = major.formula.right
                try:
                    new = modus_ponens(left, right, minor.formula)
                except (NoUnifier, TooBig):
                    continue

                if self._retain(new):
                    unprocessed.append(Entry(new, major, minor))
        self._add_to_passive(unprocessed)

    def _retain(self, new: F) -> bool:
        """check whether `new` should be retained"""
        # skip duplicates
        if new in self.seen:
            return False
        self.seen.add(new)

        # forwards subsumption
        if any(match(generalisation.formula, new) for generalisation in self.active):
            return False

        return True

    def _add_to_passive(self, new: list[Entry]):
        """add `new` to `self.passive`"""

        if new and self.model is not None:
            self.logits.extend(self.model.predict(new, self.goal).tolist())
        self.passive.extend(new)
