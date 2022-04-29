import copy
from typing import List
from searl.neuroevolution.components.evolvable_macro_network import EvolvableMacroNetwork
from searl.neuroevolution.components.cell import EvolvableMLPCell
import numpy as np

class IndividualMacro():

    def __init__(self, state_dim, action_dim, actor_config, critic_config, rl_config, index, td3_double_q,
                 critic_2_config=None, replay_memory=None, rand_init = False, micro_population = None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.rl_config = rl_config
        self.index = index
        self.td3_double_q = td3_double_q

        if critic_2_config is None:
            critic_2_config = copy.deepcopy(critic_config)

        # TODO decide how we want to initialize the network
        if rand_init:
            assert(micro_population is not None)
            cell_ids = list(micro_population.keys())
            #randomly select cell to initialize evolvable macros
            print(cell_ids)
            cell_ids_to_insert = np.random.choice(cell_ids, size=3, replace=True).tolist()

            ev_cells = []
            for cell_id in cell_ids_to_insert:
                micro_ind = micro_population[cell_id]
                ev_copy = micro_ind.cell.clone()
                micro_ind.add_cell(ev_copy)
                ev_cells.append(ev_copy)

            actor_layers = [ev_cells[0]]
            critic_1_layers = [ev_cells[1]]
            critic_2_layers = [ev_cells[2]]

            self.actor = EvolvableMacroNetwork(state_dim, num_outputs=action_dim, **actor_config,
                                               layers=actor_layers)
            self.critic_1 = EvolvableMacroNetwork(num_inputs=state_dim + action_dim, num_outputs=1, **critic_config,
                                                  layers=critic_1_layers)
            if td3_double_q:
                self.critic_2 = EvolvableMacroNetwork(num_inputs=state_dim + action_dim, num_outputs=1,
                                                      **critic_2_config, layers=critic_2_layers)

        else:
            self.actor = EvolvableMacroNetwork(num_inputs=state_dim, num_outputs=action_dim, **actor_config)
            self.critic_1 = EvolvableMacroNetwork(num_inputs=state_dim + action_dim, num_outputs=1, **critic_config)
            if td3_double_q:
                self.critic_2 = EvolvableMacroNetwork(num_inputs=state_dim + action_dim, num_outputs=1, **critic_2_config)





        self.fitness = []
        self.improvement = 0
        self.train_log = {"pre_fitness": None, "pre_rank": None, "post_fitness": None, "post_rank": None, "eval_eps": 0,
                          "index": None, "parent_index": None, "mutation": None}

        self.replay_memory = replay_memory

    def update_cell_fitnesses(self):
        self.actor.update_cell_fitnesses(self.fitness[-1])
        self.critic_1.update_cell_fitnesses(self.fitness[-1])
        if hasattr(self, 'critic_2'):
            self.critic_2.update_cell_fitnesses(self.fitness[-1])

    def update_active_population(self):
        self.actor.update_active_population()
        self.critic_1.update_active_population()
        if hasattr(self, 'critic_2'):
            self.critic_2.update_active_population()

    def get_active_population(self):
        s1 = self.actor.get_active_population()
        s2 = self.critic_1.get_active_population()
        if hasattr(self, 'critic_2'):
            s3 = self.critic_2.get_active_population()
        return s1.union(s2, s3)

    def get_cells(self):
        s1 = self.actor.get_cells()
        s2 = self.critic_1.get_cells()
        if hasattr(self, 'critic_2'):
            s3 = self.critic_2.get_cells()
        return s1.union(s2, s3)

    #returns boolean does this individual contain this cell
    def contains_cell(self, cell_id):
        assert(type(cell_id) == int)
        return cell_id in self.actor.contained_active_population or \
                cell_id in self.critic_1.contained_active_population or \
                cell_id in self.critic_2.contained_active_population

    def clone(self, index=None, copy_fitness = False):
        if index is None:
            index = self.index

        if self.td3_double_q:
            critic_2_config = copy.deepcopy(self.critic_2.short_dict)
        else:
            critic_2_config = None

        clone = type(self)(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           actor_config=copy.deepcopy(self.actor.short_dict),
                           critic_config=copy.deepcopy(self.critic_1.short_dict),
                           rl_config=copy.deepcopy(self.rl_config),
                           index=index,
                           td3_double_q=self.td3_double_q,
                           critic_2_config=critic_2_config,
                           replay_memory=self.replay_memory)

        if copy_fitness:
            clone.fitness = copy.deepcopy(self.fitness)

        clone.train_log = copy.deepcopy(self.train_log)
        clone.actor = self.actor.clone()
        clone.critic_1 = self.critic_1.clone()
        if self.td3_double_q:
            clone.critic_2 = self.critic_2.clone()

        if self.replay_memory:
            self.replay_memory = copy.deepcopy(self.replay_memory)

        return clone

    def clone_and_insert_mutated_cells(self, cell_id_to_change: int,
            actor_mutated_cells: List[EvolvableMLPCell],
            critic_1_mutated_cells: List[EvolvableMLPCell],
            critic_2_mutated_cells: List[EvolvableMLPCell],
            micro_ind_population_dict: dict, index=None, copy_fitness = False):

        if index is None:
            index = self.index

        if self.td3_double_q:
            critic_2_config = copy.deepcopy(self.critic_2.short_dict)
        else:
            critic_2_config = None

        clone = type(self)(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           actor_config=copy.deepcopy(self.actor.short_dict),
                           critic_config=copy.deepcopy(self.critic_1.short_dict),
                           rl_config=copy.deepcopy(self.rl_config),
                           index=index,
                           td3_double_q=self.td3_double_q,
                           critic_2_config=critic_2_config,
                           replay_memory=self.replay_memory)

        if copy_fitness:
            clone.fitness = copy.deepcopy(self.fitness)

        clone.train_log = copy.deepcopy(self.train_log)
        clone.actor = self.actor.clone_and_insert_mutated_cells(micro_ind_population_dict,
                                                                actor_mutated_cells, cell_id_to_change)
        clone.critic_1 = self.critic_1.clone(micro_ind_population_dict,
                                            critic_1_mutated_cells, cell_id_to_change)
        if self.td3_double_q:
            clone.critic_2 = self.critic_2.clone(micro_ind_population_dict,
                                                critic_2_mutated_cells, cell_id_to_change)

        if self.replay_memory:
            self.replay_memory = copy.deepcopy(self.replay_memory)

        return clone












