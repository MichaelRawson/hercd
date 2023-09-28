from functools import lru_cache
from typing import Union

from .constants import CACHE_SIZE, TERM_SIZE_LIMIT

F = Union[str, int, 'C']
"""a formula - constant (str), variable (int) or C"""

def name(f: F) -> str:
    """a human-readable name for a formula"""
    if isinstance(f, C):
        return repr(f)
    if isinstance(f, str):
        return f
    sign = '-' if f < 0 else ''
    f = abs(f)
    if f <= 26:
        return sign + chr(ord('a') + f - 1)
    return f'{sign}x{f}'

class TooBig(Exception):
    """a formula exceeded the size limit"""

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

    def __repr__(self) -> str:
        return f'C{name(self.left)}{name(self.right)}'

@lru_cache(maxsize=CACHE_SIZE)
def c(left: F, right: F) -> C:
    """create a C with an LRU cache"""
    return C(left, right)

def size(f: F) -> int:
    """tree size of a formula"""
    if isinstance(f, C):
        return f.size
    return 1

@lru_cache(maxsize=CACHE_SIZE)
def flip(f: F) -> F:
    """a formula with all its variables negated - useful for renaming"""
    if isinstance(f, str):
        return f
    elif isinstance(f, int):
        return -f
    else:
        return c(flip(f.left), flip(f.right))

def rename(renaming: dict[int, int], target: F) -> F:
    """canonically rename `target`, using and mutating `renaming`"""
    if isinstance(target, str):
        return target
    elif isinstance(target, int):
        try:
            return renaming[target]
        except KeyError:
            fresh = len(renaming) + 1
            renaming[target] = fresh
            return fresh
    else:
        return c(rename(renaming, target.left), rename(renaming, target.right))

@lru_cache(maxsize=CACHE_SIZE)
def substitute(variable: int, formula: F, target: F) -> F:
    """substitute `variable` for `formula` in `target`"""
    if isinstance(target, str):
        return target
    elif isinstance(target, int):
        return formula if variable == target else target
    else:
        left = substitute(variable, formula, target.left)
        right = substitute(variable, formula, target.right)
        return c(left, right)


def apply(subst: dict[int, F], target: F) -> F:
    """apply `subst` to `target`"""
    if isinstance(target, str):
        return target
    elif isinstance(target, int):
        return subst.get(target, target)
    else:
        left = apply(subst, target.left)
        right = apply(subst, target.right)
        return c(left, right)

@lru_cache(maxsize=CACHE_SIZE)
def occurs(x: int, f: F) -> bool:
    """true if x occurs in F"""
    if isinstance(f, str):
        return False
    if isinstance(f, int):
        return x == f
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
        elif isinstance(left, str):
            if left != right:
                raise NoUnifier()
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
        elif isinstance(left, str):
            if left != right:
                return False
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
