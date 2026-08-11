from __future__ import annotations
from typing import Callable, Collection, Generator, Iterable, Iterator, List, Optional, Hashable, Any, Sequence, Tuple
import pandas as pd
import itertools


class Addable:
    """A list of tuples that can be added to another list of tuples
    Each tuple contains a `(index:Hashable, value:float)`. When the lists are added elements with the same index are added.
    they merge elements with the same `index`, either adding or subtracting from the
    float value. If the float component an element becomes zero then it is removed
    from the list.

    >>> Addable([(1, 1), (1, 2), (2, 3)]) + Addable([(1, 1), (2, 3)])
    Addable([(1, 2), (2, 0)])

    Note that the list cannot be instantiated with zero tuples:

    >>> Addable([(1,0)])
    Addable([])

    """
    __slots__ = ['_data']
    _data:List[Tuple[Hashable, float]]

    def __init__(self, data:List[tuple[Hashable, float]]):
        self._data = [
            (groupby_index, the_sum)
            for groupby_index, index_overlap_iterator in
            itertools.groupby(sorted(data), lambda item:item[0])
            if abs(the_sum:=sum(item[1] for item in index_overlap_iterator)) > 1e-6
        ]
    
    def __add__(self, other:Addable) -> Addable:
        return Addable(self._data + other._data)

    def __neg__(self) -> Addable:
        return Addable([(item[0], -item[1]) for item in self._data])
    
    def __sub__(self, other:Addable) -> Addable:
        return self + (-other)
    
    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        inner = ', '.join(f'({item[0]}, {item[1]:.3f})' for item in self._data)
        return f"[{inner}]"

    def copy(self) -> Addable:
        return Addable(self._data.copy())
    
    def map(self, func:Callable[[tuple[Hashable, float]], tuple[Hashable, float]]) -> Addable:
        return Addable([func(item) for item in self._data])

    def __iter__(self) -> Iterator[tuple[Hashable, float]]:
        return iter(self._data)