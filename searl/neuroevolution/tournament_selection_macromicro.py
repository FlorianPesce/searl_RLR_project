import numpy as np


class TournamentSelection():

    def __init__(self, config):
        self.cfg = config

    def _tournament(self, fitness_values):
        selection = np.random.randint(0, len(fitness_values), size=self.cfg.nevo.tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        winner = selection[np.argmax(selection_values)]
        return winner

    def select_cell_population(self, cell_population):
        if not self.cfg.cell.population_limit:
            print("Warning, cell population not limited")
            return cell_population
        if len(cell_population) < self.cfg.cell.population_max:
            return cell_population

        select_method = self.cfg.cell.select_method
        if select_method == 'tournament':
            elite, new_cell_population = self.select_ind(cell_population)
            return new_cell_population

        elif select_method == 'least_fit':
            # remove as many inactive as possible
            # (up to percent inactive)

            # remove remaining unfit population


        #elif select_method == 'reverse_tournament':
        else:
            raise Exception: "method not implemented"






    def select_ind(self, population):
        if isinstance(population[0], EvolvableMLPCell):
            last_fitness = [cell.mean_fitness for cell in population]
            population_size = self.cfg.cell.population_max
        elif isinstance(population[0], IndividualMacro):
            last_fitness = [indi.fitness[-1] for indi in population]
            population_size = self.cfg.nevo.population_size
        else:
            raise Exception("population class unrecognized")

        #returns rank in corresponding position
        rank = np.argsort(last_fitness).argsort()

        max_id = max([ind.index for ind in population])

        #copies the best individual somehow
        elite = copy.deepcopy([population[np.argsort(rank)[-1]]][0])

        new_population = []
        if self.cfg.nevo.elitism:
            #appends best individual in population
            #no matter what
            new_population.append(elite.clone())
            selection_size = population_size - 1
        else:
            selection_size = population_size

        for idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(max_id)
            new_individual.train_log["parent_index"] = actor_parent.index
            new_population.append(new_individual)

        return elite, new_population
