from collections.abc import Generator
import random
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from .cd import C, F, TooBig, NoUnifier, match, modus_ponens
from .constants import STEP_LIMIT, EPSILON
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
            if weights is None:
                if self.model:
                    major_embeddings = torch.cat(self.major_embeddings)
                    minor_embeddings = torch.cat(self.minor_embeddings)
                    inner = major_embeddings @ minor_embeddings.T
                    probs = torch.sigmoid(inner)
                    weights = probs.view(num_choices).tolist()
                else:
                    weights = [1.0] * num_choices

            choice = random.choices(range(num_choices), weights=weights, k=1)[0]
            first = self.known[choice // len(self.known)]
            second = self.known[choice % len(self.known)]
            major = first.formula
            minor = second.formula
            if not isinstance(major, C):
                continue
            try:
                new = modus_ponens(major.left, major.right, minor)
            except (NoUnifier, TooBig):
                continue

            if self._add(new, first, second):
                weights = None
            else:
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

    def _add(self, formula: F, *parents: Entry) -> bool:
        """try to add `formula` to `known`"""
        # skip duplicates
        if formula in self.seen:
            return False
        self.seen.add(formula)

        # forwards subsumption
        for known in self.known:
            if match(known.formula, formula):
                return False

        # `formula` will be retained at this point
        if self.chatty:
            print(len(self.known), formula)
        if self.recording:
            self.log.append(parents)

        # backwards subsumption
        for index in reversed(range(len(self.known))):
            if match(formula, self.known[index].formula):
                del self.known[index]
                if self.model:
                    del self.major_embeddings[index]
                    del self.minor_embeddings[index]

        entry = Entry(
            formula,
            parents,
        )
        if match(formula, self.goal):
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
            self.major_embeddings.append(major)
            self.minor_embeddings.append(minor)
        return True
