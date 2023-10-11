from collections.abc import Generator
import random
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from .cd import C, F, TooBig, NoUnifier, match, modus_ponens
from .constants import STEP_LIMIT
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
        """this entry and all its parents, recursively"""
        yield self
        for parent in self.parents:
            yield from parent.ancestors()

    def __repr__(self):
        return repr(self.formula)

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
    """known deductions so far this episode, in chronological order"""
    major_embeddings: list[Tensor]
    """major embeddings for each entry"""
    minor_embeddings: list[Tensor]
    """minor embeddings for each entry"""
    seen: set[F]
    """inferences we've already seen this episode"""
    log: list[tuple[Entry, Entry]]
    """list of deductions made this episode, in chronological order"""
    recording: bool
    """whether we should append to `log`"""
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
        self.major_embeddings = []
        self.minor_embeddings = []
        self.seen = set()
        self.log = []
        self.proof = None
        self.recording = False
        for axiom in self.axioms:
            self._add(axiom)

    def run(self) -> bool:
        """run an episode, returning True if we found a proof"""
        self.reset()
        self.recording = True

        if self.model:
            self.model.eval()

        weights = None
        while self.proof is None and len(self.known) < STEP_LIMIT:
            num_choices = len(self.known) * len(self.known)
            if self.model and weights is None:
                major_embeddings = torch.stack(self.major_embeddings, dim=0)
                minor_embeddings = torch.stack(self.minor_embeddings, dim=1)
                inner = major_embeddings @ minor_embeddings
                probs = torch.sigmoid(inner)
                weights = probs.view(num_choices)

            if weights is None:
                choice = random.randrange(num_choices)
            else:
                choice = torch.multinomial(weights, 1)

            first = self.known[choice // len(self.known)]
            second = self.known[choice % len(self.known)]
            if self._step(first, second):
                weights = None
            elif weights is not None:
                weights[choice] = 0.0

        return self.proof is not None

    def training_graphs(self) -> Generator[tuple[Graph, Graph, bool], None, None]:
        """generate graphs for training based on the previous episode"""

        # find what we're aiming for
        target = self.proof if self.proof is not None else max(self.known, key=lambda entry: entry.tree_size)
        assert target is not None

        # all the inferences we need for `target`
        inferences = {
            entry.parents
            for entry in target.ancestors()
            if entry.parents
        }

        # all the (major, minor) pairs and whether they appear in the proof of `target`
        for first, second in self.log:
            major = Graph(first.formula, target.formula)
            minor = Graph(second.formula, target.formula)
            y = (first, second) in inferences
            yield (major, minor, y)

    def _step(self, first: Entry, second: Entry) -> bool:
        """try a CD inference with `first` and `second` and add the result to `known`"""

        major = first.formula
        minor = second.formula
        if not isinstance(major, C):
            return False
        try:
            new = modus_ponens(major.left, major.right, minor)
        except (NoUnifier, TooBig):
            return False

        return self._add(new, first, second)

    def _add(self, new: F, *parents: Entry) -> bool:
        """add `new` to `known`, recording its parents"""

        # skip duplicates
        if new in self.seen:
            return False
        self.seen.add(new)

        # forwards subsumption
        for known in self.known:
            if match(known.formula, new):
                return False

        # `formula` will be retained at this point
        if self.chatty:
            print(len(self.known), new)
        if self.recording:
            self.log.append(parents)

        # backwards subsumption
        for index in reversed(range(len(self.known))):
            if match(new, self.known[index].formula):
                del self.known[index]
                if self.model:
                    del self.major_embeddings[index]
                    del self.minor_embeddings[index]

        entry = Entry(new, parents)
        if match(new, self.goal):
            self.proof = entry
            if self.chatty:
                print("proof!")

        self.known.append(entry)
        if self.model:
            graph = Graph(entry.formula, self.goal).torch()
            batch = Batch.from_data_list([graph]).to('cuda')
            assert isinstance(batch, Data)
            with torch.no_grad():
                major, minor = self.model(batch, batch)
            self.major_embeddings.append(major.view(-1))
            self.minor_embeddings.append(minor.view(-1))
        return True
