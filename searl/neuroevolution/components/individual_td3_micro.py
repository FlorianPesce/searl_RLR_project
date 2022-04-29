from __future__ import annotations
import numpy as np


from searl.neuroevolution.components.cell import EvolvableMLPCell 

#one instance of this for all unique cells
class IndividualMicro():
    def __init__(self, id: int, cell: EvolvableMLPCell) -> None:
        # current state of the cell
        self.cell = cell
        # Dict[intra_id: EvolvableMLPCell]
        self.cell_copies_in_population = dict()
        self.cell_counter = 1 
        self.active_population = False # list of evolvable mlp cells which 
        self.mean_fitness = None
        self.improvement = 0
        self.id = id

    def set_id(self, id: int) -> None:
        self.id = id

    def set_mean_fitness(self, mean_fitness: float) -> None:
        self.mean_fitness = mean_fitness

    def add_cell(self, cell: EvolvableMLPCell) -> None:
        self.cell_copies_in_population[self.cell_counter] = cell
        cell.set_intra_id(self.cell_counter)
        cell.id = self.id
        self.cell_counter += 1

    def remove_cell_id(self, cell_id: int) -> None:
        self.cell_copies_in_population[cell_id].set_intra_id(None)
        del self.cell_copies_in_population[cell_id]

    def update_mean_fitness(self) -> float:
        self.mean_fitness = np.mean(
            [ind.fitness for ind in self.cell_copies_in_population])
        return self.mean_fitness

    def clone(self, new_id: int = None) -> IndividualMicro:
        clone = type(self)(new_id, self.cell.clone(new_id))
        for intra_id in self.cell_copies_in_population:
            cloned_cell = self.cell_copies_in_population[intra_id].clone(new_id)
            clone.add_cell(cloned_cell)

        clone.active_population = self.active_population
        clone.mean_fitness = self.mean_fitness
        clone.improvement = self.improvement

        return clone

    def clone_without_cell_copies(self, new_id: int = None) -> IndividualMicro:
        clone = type(self)(new_id, self.cell.clone())

        clone.active_population = self.active_population
        clone.mean_fitness = self.mean_fitness
        clone.improvement = self.improvement

        return clone
