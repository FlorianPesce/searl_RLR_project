import numpy as np
from __future__ import annotations

#one instance of this for all unique cells
class IndividualMicro():
    def __init__(self, id) -> None:

        # set of identical cell architectures (evolvableMlpCell)
        # set of EvolvableMLPCells
        # one for each copy of a cell
        self.cell_copies_in_population = []
        self.active_population = False #list of evolvable mlp cells which 
        self.mean_fitness = None
        self.improvement = 0
        self.id = id

    def add_cell(self, cell) -> None:
        self.cell_copies_in_population.append(cell)

    def remove_cell(self, cell) -> None:
        self.cell_copies_in_population.remove(cell)

    def update_mean_fitness(self) -> float:
        self.mean_fitness = np.mean(
            [ind.fitness for ind in self.cell_copies_in_population])
        return self.mean_fitness

    def clone(self) -> IndividualMicro:
        clone = type(self)()
        for cell in self.cell_copies_in_population:
            cloned_cell = cell.clone()
            clone.cell_copies_in_population.append(cloned_cell)

        clone.active_population = self.active_population
        clone.mean_fitness = self.mean_fitness
        clone.improvement = self.improvement

        return clone

    def clone_without_cell_copies(self) -> IndividualMicro:
        clone = type(self)()

        clone.active_population = self.active_population
        clone.mean_fitness = self.mean_fitness
        clone.improvement = self.improvement

        return clone
