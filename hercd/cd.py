from functools import lru_cache
from typing import Dict, Union, Set, Tuple

from .constants import CACHE_SIZE, TERM_SIZE_LIMIT

"""a formula - constant (str), variable (int) or C"""
F = Union[str, int, 'C']

"""a formula exceeded the size limit"""
class TooBig(Exception):
    pass

"""implication operator"""
class C:
    left: F
    right: F
    size: int
    hash: int

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
        return f'C{self.left}{self.right}'

"""create a C with an LRU cache"""
@lru_cache(maxsize=CACHE_SIZE)
def c(left: F, right: F) -> C:
    return C(left, right)

"""tree size of a formula"""
def size(f: F) -> int:
    if isinstance(f, C):
        return f.size
    return 1

@lru_cache(maxsize=CACHE_SIZE)
def negated(f: F) -> F:
    if isinstance(f, str):
        return f
    elif isinstance(f, int):
        return -f
    else:
        return c(negated(f.left), negated(f.right))

"""canonically rename `target`, using and mutating `renaming`"""
def rename(renaming: Dict[int, int], target: F) -> F:
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

"""rename `target` with an empty renaming"""
def canonical(target: F) -> F:
    return rename({}, target)

"""substitute `variable` to `formula` in `target`"""
@lru_cache(maxsize=CACHE_SIZE)
def substitute(variable: int, formula: F, target: F) -> F:
    if isinstance(target, str):
        return target
    elif isinstance(target, int):
        return formula if variable == target else target
    else:
        left = substitute(variable, formula, target.left)
        right = substitute(variable, formula, target.right)
        return c(left, right)


"""apply `subst` to `target`"""
def apply(subst: Dict[int, F], target: F) -> F:
    if isinstance(target, str):
        return target
    elif isinstance(target, int):
        return subst.get(target, target)
    else:
        left = apply(subst, target.left)
        right = apply(subst, target.right)
        return c(left, right)

"""true if x occurs in F"""
@lru_cache(maxsize=CACHE_SIZE)
def occurs(x: int, f: F) -> bool:
    if isinstance(f, str):
        return False
    if isinstance(f, int):
        return x == f
    return occurs(x, f.left) or occurs(x, f.right)

"""two formulas don't unify"""
class NoUnifier(Exception):
    pass

"""unify two formulas and return their mgu"""
def unify(left: F, right: F) -> Dict[int, F]:
    subst: Dict[int, F] = {}
    todo: Set[Tuple[F, F]] = set()

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

"""determine if `left` matches `right`"""
@lru_cache(maxsize=CACHE_SIZE)
def match(left: F, right: F) -> bool:
    match: Dict[int, F] = {}
    todo: Set[Tuple[F, F]] = set()
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

"""modus ponens: unify `antecedent` and `target`, apply mgu to `consequent` and rename"""
def modus_ponens(antecedent: F, consequent: F, target: F) -> F:
    target = negated(target)
    subst = unify(antecedent, target)
    result = apply(subst, consequent)
    return rename({}, result)
