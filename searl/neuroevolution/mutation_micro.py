import copy
from typing import List, Dict

import fastrand
import numpy as np

from searl.neuroevolution.components.cell import EvolvableMLPCell
from searl.neuroevolution.components.individual_td3_micro import IndividualMicro
#import searl.neuroevolution.searl_td3_micromacro as SEARLforTD3
import searl.neuroevolution.searl_td3_micromacro

# This file implements the mutations applied within a cell
class MicroMutations():
    def __init__(self, config, replay_sample_queue):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.mutation)
        self.replay_sample_queue = replay_sample_queue

    def no_mutation(self, individual_micro: IndividualMicro):
        individual_micro.train_log["mutation"] = "no_mutation"
        return individual_micro

    def mutation(self, population: List[IndividualMicro], searl)\
            -> Dict[int, IndividualMicro]:

        mutation_options = []
        mutation_proba = []

        #THIS IS THE SAME AS JUST MUTATING LESS CELLS, BUT IT MAKES THE CODE MORE COMPLICATED
        #JUST REDUCE THE NUMBER OF CELLS SELECTED FOR MUTATION
        #if self.cfg.micro_mutation.no_mutation:
        #    mutation_options.append(self.no_mutation)
        #    mutation_proba.append(float(self.cfg.micro_mutation.no_mutation))
        if self.cfg.micro_mutation.architecture:
            mutation_options.append(self.architecture_mutate)
            mutation_proba.append(float(self.cfg.micro_mutation.architecture))
        #if self.cfg.micro_mutation.parameters:
        #    mutation_options.append(self.parameter_mutation)
        #    mutation_proba.append(float(self.cfg.micro_mutation.parameters))
        if self.cfg.micro_mutation.activation:
            mutation_options.append(self.activation_mutation)
            mutation_proba.append(float(self.cfg.mutation.activation))
        #if self.cfg.micro_mutation.rl_hyperparam:
        #    mutation_options.append(self.rl_hyperparam_mutation)
        #    mutation_proba.append(float(self.cfg.mutation.rl_hyperparam))

        if len(mutation_options) == 0:
            return population

        mutation_proba = np.array(mutation_proba) / np.sum(mutation_proba)

        mutation_choice = self.rng.choice(mutation_options, len(population),\
                                          p=mutation_proba)

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            mutated_individual = mutation(individual)
            mutated_individual.set_id(searl.get_and_increment_ind_micro_counter())
            #mutated_population[searl.individual_micro_counter] = mutated_individual
            mutated_population.append(mutated_individual)

        return mutated_population

    def activation_mutation(self, individual_micro: IndividualMicro)\
            -> IndividualMicro:
        # TODO copy all cells then mutate
        # returns clone of individual
        # individual = individual.clone()
        # question from Florian: why clone here?
        individual_micro = individual_micro.clone()
        individual_micro = self._permutate_activation(individual_micro)
        individual_micro.train_log["mutation"] = "activation"
        return individual_micro

    #creates copies of evolvable cells, but doesn't create copies
    #of cell class, maybe update for consistency
    def _permutate_activation(self, individual_micro: IndividualMicro)\
            -> IndividualMicro:
        first_cell = individual_micro.cell_copies_in_population[0]
        possible_activations = ['relu', 'elu', 'tanh']
        current_activation = first_cell.activation
        possible_activations.remove(current_activation)
        new_activation = self.rng.choice(possible_activations, size=1)[0]
        
        for cell in [individual_micro.cell]\
                + individual_micro.cell_copies_in_population:
            net_dict = cell.init_dict
            net_dict['activation'] = new_activation
            new_cell = type(cell)(**net_dict)
            new_cell.load_state_dict(cell.state_dict())
            cell = new_cell

        return individual_micro
    
    #creates a copy of everything before mutation
    #but I'm pretty sure we can just change this to not create a copy
    #and create the copy in searl td3
    def architecture_mutate(self, individual_micro: IndividualMicro)\
            -> IndividualMicro:
        #copies individual and all cells
        offspring_individual_micro = individual_micro.clone()
        rand_numb = self.rng.uniform(0, 1)
        if rand_numb < self.cfg.mutation.new_layer_prob:
            for cell in [offspring_individual_micro.cell]\
                    + offspring_individual_micro.cell_copies_in_production:
                cell.add_layer()
            individual_micro.train_log["mutation"] = "architecture_new_layer"
        else:
            node_dict = offspring_individual_micro\
                                    .cell_copies_in_production[0].add_node()
            for cell in [offspring_individual_micro.cell]\
                    + offspring_individual_micro.cell_copies_in_production[1:]:
                cell.add_node(**node_dict)
            individual_micro.train_log["mutation"] = "architecture_new_node"

        individual_micro = offspring_individual_micro
        return individual_micro
