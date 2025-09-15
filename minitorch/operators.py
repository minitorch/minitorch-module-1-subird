"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function: return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y. Returns 1.0 if true, 0.0 if false."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x equals y. Returns 1.0 if true, 0.0 if false."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close (within 1e-2). Returns 1.0 if true, 0.0 if false."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function with numerical stability."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply ReLU activation function: max(0, x)."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculate the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Compute the derivative of log(x) times a second argument y."""
    return y / x


def inv(x: float) -> float:
    """Calculate the reciprocal (1/x)."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Compute the derivative of reciprocal(x) times a second argument y."""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Compute the derivative of ReLU(x) times a second argument y."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a function to each element of an iterable."""
    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines two iterables element-wise using a function."""
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return _zipWith


def reduce(
    fn: Callable[[float, float], float], default: Optional[float] = None
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value starting with the first element.

    Args:
    ----
        fn: Function to apply for reduction
        default: Optional value to return for empty sequences. If None, raises ValueError for empty sequences.
    """
    def _reduce(ls: Iterable[float]) -> float:
        items = list(ls)
        if not items:
            if default is not None:
                return default
            raise ValueError("reduce of empty sequence")
        result = items[0]
        for x in items[1:]:
            result = fn(result, x)
        return result
    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list. Returns 0.0 for empty lists."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list. Returns 1.0 for empty lists."""
    return reduce(mul, 1.0)(ls)
