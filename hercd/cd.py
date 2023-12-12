from functools import lru_cache
from typing import Generator, Optional, Union

from .constants import CACHE_SIZE, TERM_SIZE_LIMIT

F = Union[int, 'N', 'C']
"""a formula - constant (str), variable (int) or C"""

def name(f: F) -> str:
    """a human-readable name for a formula"""
    if not isinstance(f, int):
        return str(f)

    sign = '-' if f < 0 else ''
    f = abs(f)
    if f > 26:
        return f'{sign}x{f}'

    return sign + chr(ord('a') + f - 1)

class TooBig(Exception):
    """a formula exceeded the size limit"""


class N:
    """a 'N' (negation) formula"""

    negated: F
    """the negated formula"""
    size: int
    """the tree size of the formula"""
    hash: int
    """a precomputed hash"""

    def __init__(self, negated: F):
        assert negated != 0
        self.negated = negated
        self.size = size(negated) + 1
        if self.size > TERM_SIZE_LIMIT:
            raise TooBig()
        self.hash = hash((negated,))

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: F) -> bool:
        return id(self) == id(other) or isinstance(other, N) and self.negated == other.negated

    def __str__(self) -> str:
        return f'N{name(self.negated)}'

    def __repr__(self) -> str:
        return f'n({repr(self.negated)})'

@lru_cache(maxsize=CACHE_SIZE)
def n(negated: F) -> N:
    """create a N with an LRU cache"""
    return N(negated)


class C:
    """a 'C' (implication) formula"""

    left: F
    """the left-hand side"""
    right: F
    """the right-hand side"""
    size: int
    """the tree size of the formula"""
    hash: int
    """a precomputed hash"""

    def __init__(self, left: F, right: F):
        assert left != 0 and right != 0
        self.left = left
        self.right = right
        self.size = size(left) + size(right) + 1
        if self.size > TERM_SIZE_LIMIT:
            raise TooBig()
        self.hash = hash((left, right))

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: F) -> bool:
        return id(self) == id(other) or isinstance(other, C) and self.left == other.left and self.right == other.right

    def __str__(self) -> str:
        return f'C{name(self.left)}{name(self.right)}'

    def __repr__(self) -> str:
        return f'c({repr(self.left)},{repr(self.right)})'

@lru_cache(maxsize=CACHE_SIZE)
def c(left: F, right: F) -> C:
    """create a C with an LRU cache"""
    return C(left, right)


def size(f: F) -> int:
    """tree size of a formula"""
    if isinstance(f, int):
        return 1

    return f.size


@lru_cache(maxsize=CACHE_SIZE)
def flip(f: F) -> F:
    """a formula with all its variables negated - useful for renaming"""
    if isinstance(f, int):
        return -f
    elif isinstance(f, N):
        return n(flip(f.negated))
    else:
        return c(flip(f.left), flip(f.right))

def rename(renaming: dict[int, int], target: F) -> F:
    """canonically rename `target`, using and mutating `renaming`"""
    if isinstance(target, int):
        try:
            return renaming[target]
        except KeyError:
            fresh = len(renaming) + 1
            renaming[target] = fresh
            return fresh
    elif isinstance(target, N):
        return n(rename(renaming, target.negated))
    else:
        return c(rename(renaming, target.left), rename(renaming, target.right))

@lru_cache(maxsize=CACHE_SIZE)
def substitute(variable: int, formula: F, target: F) -> F:
    """substitute `variable` for `formula` in `target`"""
    if isinstance(target, int):
        return formula if variable == target else target
    elif isinstance(target, N):
        return n(substitute(variable, formula, target.negated))

    left = substitute(variable, formula, target.left)
    right = substitute(variable, formula, target.right)
    return c(left, right)


def apply(subst: dict[int, F], target: F) -> F:
    """apply `subst` to `target`"""
    if isinstance(target, int):
        return subst.get(target, target)
    elif isinstance(target, N):
        return n(apply(subst, target.negated))

    left = apply(subst, target.left)
    right = apply(subst, target.right)
    return c(left, right)

@lru_cache(maxsize=CACHE_SIZE)
def occurs(x: int, f: F) -> bool:
    """true if x occurs in F"""
    if isinstance(f, int):
        return x == f
    elif isinstance(f, N):
        return occurs(x, f.negated)

    return occurs(x, f.left) or occurs(x, f.right)

class NoUnifier(Exception):
    """two formulas don't unify"""

def unify(left: F, right: F) -> dict[int, F]:
    """unify two formulas and return their most general unifier"""
    subst: dict[int, F] = {}
    todo: set[tuple[F, F]] = set()

    todo.add((left, right))
    while todo:
        left, right = todo.pop()
        if left in subst:
            left = subst[left]
        if right in subst:
            right = subst[right]
        if left == right:
            continue

        if isinstance(left, int):
            right = apply(subst, right)
            if occurs(left, right):
                raise NoUnifier()
            for var in subst:
                subst[var] = substitute(left, right, subst[var])
            subst[left] = right
        elif isinstance(right, int):
            todo.add((right, left))
        elif isinstance(left, N):
            if not isinstance(right, N):
                raise NoUnifier()
            todo.add((left.negated, right.negated))
        elif not isinstance(right, C):
            raise NoUnifier()
        else:
            todo.add((left.left, right.left))
            todo.add((left.right, right.right))

    return subst

@lru_cache(maxsize=CACHE_SIZE)
def match(left: F, right: F) -> bool:
    """determine if `left` matches `right`"""
    match: dict[int, F] = {}
    todo: set[tuple[F, F]] = set()
    todo.add((left, right))
    while todo:
        left, right = todo.pop()
        if isinstance(left, int):
            if left in match:
                if match[left] != right:
                    return False
            else:
                match[left] = right
        elif isinstance(left, N):
            if not isinstance(right, N):
                return False
            todo.add((left.negated, right.negated))
        elif not isinstance(right, C):
            return False
        else:
            todo.add((left.left, right.left))
            todo.add((left.right, right.right))

    return True

def modus_ponens(antecedent: F, consequent: F, target: F) -> F:
    """modus ponens: unify `antecedent` and `target`, return substituted `consequent` - all with renaming"""
    target = flip(target)
    subst = unify(antecedent, target)
    result = apply(subst, consequent)
    return rename({}, result)

class Entry:
    """a deduced formula and its proof"""

    formula: F
    """the formula that this entry proves"""
    parents: tuple['Entry', ...]
    """premises - empty for axioms"""
    ancestors: set['Entry']
    """transitive relation of parents"""
    n_simplify: bool
    """is the formula a candidate for n-simplification?"""
    score: float
    """a score assigned by a neural network, if applicable"""

    def __init__(self, term: F, *parents: 'Entry'):
        self.formula = term
        self.parents = parents
        self.ancestors = set(parents).union(*(parent.ancestors for parent in parents))
        self.n_simplify = False
        self.score = 0.0
        if isinstance(self.formula, C):
            left = self.formula.left
            right = self.formula.right
            self.n_simplify = isinstance(left, int) and not occurs(left, right)

def n_simplify(major: Entry) -> F:
    """n-simplification: if x does not occur in F, then CxF => F"""
    assert major.n_simplify and isinstance(major.formula, C)
    return rename({}, major.formula.right)


def D(major: Entry, minor: Optional[Entry] = None) -> Entry:
    """manually write a proof as D-terms"""
    assert isinstance(major.formula, C)

    if not minor:
        assert major.n_simplify
        return Entry(n_simplify(major), major)
    new = modus_ponens(major.formula.left, major.formula.right, minor.formula)
    return Entry(new, major, minor)
