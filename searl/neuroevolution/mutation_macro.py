from typing import List, Optional, Dict

import fastrand
import numpy as np

from searl.neuroevolution.components.individual_td3_macro import IndividualMacro
from searl.neuroevolution.components.individual_td3_micro import IndividualMicro
from searl.neuroevolution.components.evolvable_macro_network import EvolvableMacroNetwork

# This file implements the mutations applied to an individual in the population. (Mainly adding or removing cells)
class MacroMutations():

    def __init__(self, config, replay_sample_queue):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.mutation)
        self.replay_sample_queue = replay_sample_queue

    def no_mutation(self, individual: IndividualMacro):
        individual.train_log["mutation"] = "no_mutation"
        return individual

    def mutation(self, population: List[IndividualMacro],
            micro_population: Dict[int, IndividualMicro] = None)\
            -> List[IndividualMacro]:

        mutation_options = []
        mutation_proba = []
        if self.cfg.mutation.no_mutation:
            mutation_options.append(self.no_mutation)
            mutation_proba.append(float(self.cfg.mutation.no_mutation))
        if self.cfg.mutation.architecture:
            mutation_options.append(self.architecture_mutate)
            mutation_proba.append(float(self.cfg.mutation.architecture))
        if self.cfg.mutation.parameters:
            mutation_options.append(self.parameter_mutation)
            mutation_proba.append(float(self.cfg.mutation.parameters))
        if self.cfg.mutation.activation:
            mutation_options.append(self.activation_mutation)
            mutation_proba.append(float(self.cfg.mutation.activation))
        if self.cfg.mutation.rl_hyperparam:
            mutation_options.append(self.rl_hyperparam_mutation)
            mutation_proba.append(float(self.cfg.mutation.rl_hyperparam))

        if len(mutation_options) == 0:
            return population

        mutation_proba = np.array(mutation_proba) / np.sum(mutation_proba)

        mutation_choice = self.rng.choice(mutation_options, len(population), p=mutation_proba)

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            if mutation == self.architecture_mutate:
                mutated_population.append(mutation(individual, micro_population))
            else:
                mutated_population.append(mutation(individual))

        return mutated_population

    def rl_hyperparam_mutation(self, individual: IndividualMacro):

        rl_config = individual.rl_config
        rl_params = self.cfg.mutation.rl_hp_selection
        mutate_param = self.rng.choice(rl_params, 1)[0]

        random_num = self.rng.uniform(0, 1)
        if mutate_param == 'train_frames_fraction':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(0.1, max(3.0, getattr(rl_config, mutate_param) * 1.2)))
            else:
                setattr(rl_config, mutate_param, min(0.1, max(3.0, getattr(rl_config, mutate_param) * 0.8)))
        elif mutate_param == 'batch_size':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(128, max(8, int(getattr(rl_config, mutate_param) * 1.2))))
            else:
                setattr(rl_config, mutate_param, min(128, max(8, int(getattr(rl_config, mutate_param) * 0.8))))
        elif mutate_param == 'lr_actor':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 1.2)))
            else:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 0.8)))
        elif mutate_param == 'lr_critic':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 1.2)))
            else:
                setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 0.8)))
        elif mutate_param == 'td3_policy_noise':
            if getattr(rl_config, mutate_param):
                setattr(rl_config, mutate_param, False)
            else:
                setattr(rl_config, mutate_param, 0.1)
        elif mutate_param == 'td3_update_freq':
            if random_num > 0.5:
                setattr(rl_config, mutate_param, min(10, max(1, int(getattr(rl_config, mutate_param) + 1))))
            else:
                setattr(rl_config, mutate_param, min(10, max(1, int(getattr(rl_config, mutate_param) - 1))))
        elif mutate_param == 'optimizer':
            opti_selection = ["adam", "adamax", "rmsprop", "sdg"]
            opti_selection.remove(getattr(rl_config, mutate_param))
            opti = self.rng.choice(opti_selection, 1)
            setattr(rl_config, mutate_param, opti)

        individual.train_log["mutation"] = "rl_" + mutate_param
        individual.rl_config = rl_config
        return individual

    def activation_mutation(self, individual: IndividualMacro):
        individual.actor = self._permutate_activation(individual.actor)
        individual.critic_1 = self._permutate_activation(individual.critic_1)
        if self.cfg.train.td3_double_q:
            individual.critic_2 = self._permutate_activation(individual.critic_2)
        individual.train_log["mutation"] = "activation"
        return individual

    def _permutate_activation(self, network: EvolvableMacroNetwork):

        possible_activations = ['relu', 'elu', 'tanh']
        current_activation = network.activation
        possible_activations.remove(current_activation)
        new_activation = self.rng.choice(possible_activations, size=1)[0]
        net_dict = network.init_dict
        net_dict['activation'] = new_activation
        new_network = type(network)(**net_dict)
        new_network.load_state_dict(network.state_dict())
        network = new_network

        return network

    def parameter_mutation(self, individual: IndividualMacro):

        offspring = individual.actor

        offspring = self.classic_parameter_mutation(offspring)
        individual.train_log["mutation"] = "classic_parameter"

        individual.actor = offspring
        return individual

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def classic_parameter_mutation(self, network: EvolvableMacroNetwork):
        mut_strength = self.cfg.mutation.mutation_sd
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        model_params = network.state_dict()

        potential_keys = []
        for i, key in enumerate(model_params):  # Mutate each param
            if not 'norm' in key:
                W = model_params[key]
                if len(W.shape) == 2:  # Weights, no bias
                    potential_keys.append(key)

        how_many = np.random.randint(1, len(potential_keys) + 1, 1)[0]
        chosen_keys = np.random.choice(potential_keys, how_many, replace=False)

        for key in chosen_keys:
            # References to the variable keys
            W = model_params[key]
            num_weights = W.shape[0] * W.shape[1]
            # Number of mutation instances
            num_mutations = fastrand.pcg32bounded(int(np.ceil(num_mutation_frac * num_weights)))
            for _ in range(num_mutations):
                ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                random_num = self.rng.uniform(0, 1)

                if random_num < super_mut_prob:  # Super Mutation probability
                    W[ind_dim1, ind_dim2] += self.rng.normal(0, np.abs(super_mut_strength * W[ind_dim1, ind_dim2]))
                elif random_num < reset_prob:  # Reset probability
                    W[ind_dim1, ind_dim2] = self.rng.normal(0, 1)
                else:  # mutauion even normal
                    W[ind_dim1, ind_dim2] += self.rng.normal(0, np.abs(mut_strength * W[ind_dim1, ind_dim2]))

                # Regularization hard limit
                W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], 1000000)
        return network

    def architecture_mutate(self, individual: IndividualMacro,
                            micro_population: Dict[int, IndividualMicro]):

        offspring_actor = individual.actor.clone()
        offspring_critic_1 = individual.critic_1.clone()
        if self.cfg.train.td3_double_q:
            offspring_critic_2 = individual.critic_2.clone()
        
        rand_numb = self.rng.uniform(0, 1)
        if rand_numb < self.cfg.macro_mutation.new_layer_prob:
            offspring_actor.add_layer(micro_population, insertion_method='random')
            offspring_critic_1.add_layer(micro_population, insertion_method='random')
            if self.cfg.train.td3_double_q:
                offspring_critic_2.add_layer(micro_population, insertion_method='random')
            individual.train_log["mutation"] = "architecture_new_macrolayer"
        else:
            offspring_actor.add_cell(micro_population=micro_population)
            offspring_critic_1.add_cell(micro_population=micro_population)
            if self.cfg.train.td3_double_q:
                offspring_critic_2.add_cell(micro_population=micro_population)
            individual.train_log["mutation"] = "architecture_new_macrolayer"

        individual.actor = offspring_actor
        individual.critic_1 = offspring_critic_1
        if self.cfg.train.td3_double_q:
            individual.critic_2 = offspring_critic_2
        return individual
