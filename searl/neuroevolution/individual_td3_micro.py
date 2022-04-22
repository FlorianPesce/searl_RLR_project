
#one instance of this for all unique cells
class IndividualMacro():

    def init...


        # list of identical cell architectures (evolvableMlpCell)
        #list of EvolvableMLPCells
        #one for each copy of a cell
        self.cell_copies_in_population

        self.active_population = False
        self.mean_fitness = None

        self.fitness = []
        self.improvement = 0
        self.unique_cell_id = #function of layers to ensure multiple copies of same cell are not created
