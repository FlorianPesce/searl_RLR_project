import numpy as np

#one instance of this for all unique cells
class IndividualMicro():
    def __init__(self) -> None:

        # list of identical cell architectures (evolvableMlpCell)
        #list of EvolvableMLPCells
        #one for each copy of a cell
        self.cell_copies_in_population = set()
        self.active_population = False #list of evolvable mlp cells which 
        self.mean_fitness = None
        self.improvement = 0

    def add_cell(self, cell) -> None:
        self.cell_copies_in_population.add(cell)

    def remove_cell(self, cell) -> None:
        self.cell_copies_in_population.remove(cell)

    def update_mean_fitness(self) -> float:
        self.mean_fitness = np.mean(
            [ind.fitness for ind in self.cell_copies_in_population])
        return self.mean_fitness
