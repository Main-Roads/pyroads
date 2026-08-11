from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

from .ImmutableTree import ImmutableTree


@dataclass
class Transition:
    measure_true_from:float
    measure_true_to:float
    measure_slk_from:float
    measure_slk_to:float
    tree:ImmutableTree

    def copy(self) -> Transition:
        return Transition(
            measure_true_from=self.measure_true_from,
            measure_true_to=self.measure_true_to,
            measure_slk_from=self.measure_slk_from,
            measure_slk_to=self.measure_slk_to,
            tree=self.tree.deep_clone()
        )
    
    def map_tree(self, func:Callable[[tuple[Hashable, float]], tuple[Hashable, float]]) -> Transition:
        return Transition(
            measure_true_from=self.measure_true_from,
            measure_true_to=self.measure_true_to,
            measure_slk_from=self.measure_slk_from,
            measure_slk_to=self.measure_slk_to,
            tree=self.tree.map(func)
        )
    
    def accumulate(self, next_transition:Transition) -> Transition:
        """
        ## Raises
        
        `ImmutableTreeMergeError`
            If the trees are not mergeable.
        """
        return Transition(
            measure_true_from = self.measure_true_from,
            measure_true_to   = next_transition.measure_true_to,
            measure_slk_from  = self.measure_slk_from,
            measure_slk_to    = next_transition.measure_slk_to,
            tree              = self.tree.merge(next_transition.tree)
        )
    
    def __repr__(self) -> str:
        return (
              f"Transition("
            + f"true: {self.measure_true_from:.3f} {self.measure_true_to:.3f} "
            + f"slk: {self.measure_slk_from:.3f} {self.measure_slk_to:.3f} "
            + f"tree: {self.tree} )"
        )
