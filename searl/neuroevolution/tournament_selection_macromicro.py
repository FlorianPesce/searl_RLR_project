import copy
from itertools import compress
from typing import Dict, List

import numpy as np
from searl.neuroevolution.components.cell import EvolvableMLPCell
from searl.neuroevolution.components.individual_td3_macro import \
    IndividualMacro
from searl.neuroevolution.components.individual_td3_micro import \
    IndividualMicro


class TournamentSelection():

    def __init__(self, config):
        self.cfg = config

    def _tournament(self, fitness_values, tournament_size):
        selection = np.random.randint(0, len(fitness_values), size=tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        winner = selection[np.argmax(selection_values)]
        return winner

    def _reverse_tournament(self, fitness_values, tournament_size):
        selection = np.random.randint(0, len(fitness_values), size=tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        winner = selection[np.argmin(selection_values)]
        return winner

    #def least_fit(self, population, percentage):

    '''
    def select_cell_population(self, cell_population: List[EvolvableMLPCell],\
            individual_population: List[IndividualMacro]):
        cell_population_max = self.cfg.selection.population_max
        if not self.selection.population_limit:
            print("Warning, cell population not limited")
            return cell_population
        if len(cell_population) < cell_population_max:
            return cell_population

        select_method = self.cfg.selection.select_method

        if select_method == 'tournament':
            _, new_cell_population = self.select_cell(cell_population)
            return new_cell_population

        elif select_method == 'least_fit':
            
            n_cell_pop = len(cell_population)
            n_cells_to_remove = n_cell_pop - cell_population_max

            # (up to percent inactive)
            for cell in cell_population:
                cell.active_population = False

            for ind in individual_population:
                ind.update_active_population()


            percent_inactive = self.cfg.selection.percent_inactive
            percent_inferior = self.cfg.selection.percent_inferior
            assert(percent_inferior + percent_inactive == 100)

            # remove as many inactive as possible
            max_inactive_to_remove = round((percent_inactive / 100) * n_cells_to_remove)
            inactive_cell_indices = []
            for i, cell in enumerate(cell_population):
                inactive_cell_indices.append(i)

            # select random subset of cells
            indices_to_remove = np.random.choice(np.array(inactive_cell_indices),
                            size = max_inactive_to_remove, replace = False).tolist()

            # removes cells from population
            for index in sorted(indices_to_remove, reverse=True):
                del cell_population[index]

            # remove remaining unfit population using reverse tournament selection
            n_cell_pop = len(cell_population)
            n_cells_to_remove = n_cell_pop - cell_population_max
            if n_cells_to_remove > 0:
                cell_population = self.remove_ind(cell_population, n_cells_to_remove)
                return cell_population
            else:
                return cell_population
    
        

        #elif select_method == 'reverse_tournament':
        else:
            raise Exception("method not implemented")
    '''
    def purge_dead_cells(self, micro_population: Dict[int, IndividualMicro],\
            dead_macro_population: List[IndividualMacro]) -> None:
        for dead_individual_macro in dead_macro_population:
            if dead_individual_macro.td3_double_q:
                list_networks = [dead_individual_macro.actor,
                                 dead_individual_macro.critic_1,
                                 dead_individual_macro.critic_2]
            else:
                list_networks = [dead_individual_macro.actor,
                                 dead_individual_macro.critic_1]
            for network in list_networks:
                for layer in network.layers:
                    for cell in layer.cells:
                        micro_population[cell.id].remove_cell_id(cell.intra_id)


    def select_ind_macro(self, macro_population: List[IndividualMacro],\
            micro_population: Dict[int, IndividualMicro]):
        losers = [True for x in macro_population]
        last_fitness = [indi.fitness[-1] for indi in macro_population]
        population_size = self.cfg.nevo.population_size
        tournament_size = self.cfg.nevo.tournament_size

        #returns rank in corresponding position
        rank = np.argsort(last_fitness).argsort()

        max_id = max([ind.index for ind in macro_population])

        #copies the best individual somehow
        elite = copy.deepcopy([macro_population[np.argsort(rank)[-1]]][0])

        new_population = []
        if self.cfg.nevo.elitism:
            #appends best individual in population
            #no matter what
            new_population.append(elite.clone(micro_population))
            selection_size = population_size - 1
        else:
            selection_size = population_size

        for idx in range(selection_size):
            max_id += 1
            winner = self._tournament(rank, tournament_size)
            losers[winner] = False
            actor_parent = macro_population[winner]
            new_individual = actor_parent.clone(micro_population)
            new_individual.train_log["parent_index"] = actor_parent.index
            new_population.append(new_individual)

        dead_macro_population = list(compress(macro_population, losers))
        self.purge_dead_cells(micro_population, dead_macro_population)

        return elite, new_population

    '''
    def select_cell(self, population: List[EvolvableMLPCell]):

        last_fitness = [cell.mean_fitness for cell in population]
        population_size = self.cfg.selection.population_max
        # TODO is it the right tournament size being picked here?
        tournament_size = self.cfg.nevo.tournament_size

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
            actor_parent = population[self._tournament(rank, tournament_size)]
            new_individual = actor_parent.clone(max_id)
            new_individual.train_log["parent_index"] = actor_parent.index
            new_population.append(new_individual)

        return elite, new_population
    '''

    #remove using reverse tournament selection
    def remove_ind(self, population, n_remove):
        if isinstance(population[0], EvolvableMLPCell):
            last_fitness = [cell.mean_fitness for cell in population]
            tournament_size = self.cfg.nevo.tournament_size
        elif isinstance(population[0], IndividualMacro):
            last_fitness = [indi.fitness[-1] for indi in population]
            tournament_size = self.cfg.nevo.tournament_size
        else:
            raise Exception("population class unrecognized")

        #returns rank in corresponding position
        rank = np.argsort(last_fitness).argsort()

        for idx in range(n_remove):
            #remove population member
            del population[self._reverse_tournament(rank, tournament_size)]

        return population

    