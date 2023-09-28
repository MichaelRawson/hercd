from collections.abc import Generator
import random
from typing import Optional

import torch

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
    """policy network - if None, apply a uniform policy"""
    silent: bool
    """whether to be chatty or not"""
    known: list[Entry]
    """known deductions so far this episode, in chronological order"""
    seen: set[F]
    """formulas we've already seen this episode"""
    log: list[tuple[F, tuple[Entry]]]
    """list of deductions made this episode, in chronological order"""
    recording: bool
    """whether we should append to `log`"""
    proof: Optional[Entry]
    """if not None, a proof of `goal` from `axioms`"""

    def __init__(self, axioms: list[F], goal: F):
        self.axioms = axioms
        self.goal = goal
        self.model = None
        self.silent = False
        self.reset()

    def reset(self):
        """reinitialise the environment, copying axioms to `known`"""
        self.known = []
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

        prediction = None
        while self.proof is None and len(self.known) < STEP_LIMIT:
            if self.model and random.random() > EPSILON:
                if prediction is None:
                    graph = self.graph(self.goal).torch().to('cuda')
                    with torch.no_grad():
                        prediction = torch.sigmoid(self.model(graph))
                weights = prediction.view(len(self.known) * len(self.known)).tolist()
                choice = random.choices(range(len(self.known) * len(self.known)), weights=weights, k=1)[0]
                first = self.known[choice // len(self.known)]
                second = self.known[choice % len(self.known)]
            else:
                first, second = random.choices(self.known, k=2)

            major = first.formula
            minor = second.formula
            if not isinstance(major, C):
                continue
            try:
                new = modus_ponens(major.left, major.right, minor)
            except (NoUnifier, TooBig):
                continue

            if self._add(new, first, second):
                prediction = None

        return self.proof is not None

    def training_graphs(self) -> Generator[Graph, None, None]:
        """generate graphs for training based on the previous episode"""

        # find what we're aiming for
        target = self.proof if self.proof is not None else max(self.known, key=lambda entry: entry.tree_size)
        assert target is not None

        # all the _inferences_ we need for `target`
        inferences = {
            entry.parents
            for entry in target.ancestors()
            if entry.parents
        }

        # copy self.log because it's about to be clobbered
        log = self.log
        self.reset()

        for formula, parents in log:
            # get the current graph
            graph = self.graph(target.formula)

            # work out which entries in `inferences` exist yet
            formula2index = {
                entry.formula: index
                for index, entry in enumerate(self.known)
            }
            # add the available pairs to the graph
            for first, second in inferences:
                major = first.formula
                minor = second.formula
                if major in formula2index and minor in formula2index:
                    graph.pairs.append((formula2index[major], formula2index[minor]))

            yield graph
            # add the formula only after so we get the state _before_ we did something
            self._add(formula, *parents)

    def graph(self, target: F) -> Graph:
        """generate a graph representing the current state"""
        graph = Graph(target)
        for entry in self.known:
            graph.entry(entry.formula)
        return graph

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
        if not self.silent:
            print(len(self.known), formula)
        if self.recording:
            self.log.append((formula, parents))

        # backwards subsumption
        for index in reversed(range(len(self.known))):
            if match(formula, self.known[index].formula):
                del self.known[index]

        entry = Entry(
            formula,
            parents,
        )
        if match(formula, self.goal):
            self.proof = entry
            if not self.silent:
                print("proof!")

        self.known.append(entry)
        return True
