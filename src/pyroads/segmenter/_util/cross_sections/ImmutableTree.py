from __future__ import annotations

from typing import Callable, Generator, Hashable, Optional
from .Addable import Addable

import pandas as pd


class ImmutableTreeMergeError(Exception):
    pass

class ImmutableTree:
    """
    This is not a very clever implementation of a Tree storage structure.
    It was cobbled together while itterating on the algorithm.
    It is intended to be replaced with a refactored version.
    """
    __slots__ = (
        '_children',
        '_data',
    )

    #_parent:Optional[ImmutableTree]
    _children:dict[str, ImmutableTree]
    _data:Addable
    

    def __init__(self):
        self._children = {}
        self._data     = Addable([])


    def is_leaf(self) -> bool:
        return len(self._children) == 0


    def is_empty_leaf(self) -> bool:
        return len(self._children) == 0 and len(self._data) == 0

    def deep_clone(self) -> ImmutableTree:
        clone = ImmutableTree()
        clone._data = self._data.copy()
        for key, value in self._children.items():
            clone._children[key] = value.deep_clone()
        return clone

    def prune(self) -> Optional[ImmutableTree]:
        result = self.deep_clone()
        result._children = {
            child_name:child_pruned 
            for child_name, child in result._children.items() 
            if (child_pruned:=child.prune()) is not None
        }
        if len(result._children) == 0 and len(result._data) == 0:
            return None
        return result

    def add_data(self, path_values:list[str], data:Addable) -> ImmutableTree:
        result = self.deep_clone()
        pointer = result
        for path_value in path_values:
            if path_value not in pointer._children:
                pointer._children[path_value] = ImmutableTree()
            pointer = pointer._children[path_value]
        pointer._data += data
        return result_pruned if (result_pruned:=result.prune()) is not None else ImmutableTree()
    

    def remove_data(self, node_values:list[str], data:Addable) -> ImmutableTree:
        return self.add_data(node_values, -data)

    
    def merge(self, other:ImmutableTree) -> ImmutableTree:
        
        if self.is_leaf() and other.is_leaf():
            result = self.deep_clone()
            result._data += other._data
            return result
        
        result = self.deep_clone()

        if not all(key in other._children for key in self._children):
            raise ImmutableTreeMergeError(f"Cannot merge trees with different node sets: {self} {other}")

        for key, value in other._children.items():
            if key in result._children:
                result._children[key] = result._children[key].merge(value)
            else:
                raise ImmutableTreeMergeError(f"{key} not in ImmutableTree {self}")
        return result


    def map(self, func:Callable[[tuple[Hashable, float]], tuple[Hashable, float]]) -> ImmutableTree:
        result = self.deep_clone()
        result._data = self._data.map(func)
        for key, value in self._children.items():
            result._children[key] = value.map(func)
        return result


    def __str__(self):
        #data = ' '.join(str(item) for item in self._data) if len(self._data) > 0 else ''
        data = repr(self._data)[1:-1] if len(self._data) > 0 else ''
        children = " ".join(f"{key}:{value}" for key,value in self._children.items())
        joiner = ":" if len(children) > 0 and len(data)>0 else ''
        return f"<{data}{joiner}{children}>"
    

    def __repr__(self):
        return self.__str__()


    def iter_leaf_data(self, child_name_path:Optional[list[str]]=None) -> Generator[tuple[list[str], Addable], None, None]:
        child_name_path = [] if child_name_path is None else child_name_path
        if self.is_leaf():
            yield child_name_path, self._data
        else:
            for child_name, child in self._children.items():
                yield from child.iter_leaf_data([*child_name_path, child_name])


    def get_row_data(self, child_name_path:Optional[list[str]]=None):
        child_name_path = [] if child_name_path is None else child_name_path
        if self.is_leaf():
            for item in self._data:
                yield child_name_path, item
        else:
            for child_name, child in self._children.items():
                yield from child.get_row_data([child_name, *child_name_path])


    def get_depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(child.get_depth() for child in self._children.values())


    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for data in self.get_row_data():
            rows.append([*data[0], *data[1]])
        columns = [f"L{i}" for i in range(self.get_depth())]
        return pd.DataFrame(
            data=rows
        )
