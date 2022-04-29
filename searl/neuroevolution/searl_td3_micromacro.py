import copy
import time
from typing import Dict, List, Set

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
from searl.neuroevolution.components.cell import EvolvableMLPCell
from searl.neuroevolution.components.individual_td3_macro import \
    IndividualMacro
from searl.neuroevolution.components.individual_td3_micro import \
    IndividualMicro
from searl.neuroevolution.components.replay_memory import (MPReplayMemory,
                                                           ReplayMemory)
from searl.neuroevolution.evaluation_td3 import MPEvaluation
from searl.neuroevolution.mutation_macro import MacroMutations
from searl.neuroevolution.mutation_micro import MicroMutations
from searl.neuroevolution.tournament_selection_macromicro import TournamentSelection
from searl.neuroevolution.training_td3 import TD3Training
from searl.utils.supporter import Supporter


class SEARLforTD3():

    def __init__(self, config, logger, checkpoint):

        self.cfg = config
        self.log = logger
        self.ckp = checkpoint

        torch.manual_seed(self.cfg.seed.torch)
        np.random.seed(self.cfg.seed.numpy)

        self.log.print_config(self.cfg)
        self.log.csv.fieldnames(
            ["epoch", "time_string", "eval_eps", "pre_fitness", "pre_rank",
             "post_fitness", "post_rank", "index", "parent_index", "mutation",
             "train_iterations", ] + list(self.cfg.rl.get_dict.keys()))

        self.log.log("initialize replay memory")
        if self.cfg.nevo.ind_memory:
            push_queue = None
            sample_queue = None
        else:
            self.replay_memory = \
                MPReplayMemory(seed=self.cfg.seed.replay_memory,
                               capacity=self.cfg.train.replay_memory_size,
                               batch_size=self.cfg.rl.batch_size,
                               reuse_batch=self.cfg.nevo.reuse_batch)
            push_queue = self.replay_memory.get_push_queue()
            sample_queue = self.replay_memory.get_sample_queue()

        self.eval = MPEvaluation(config=self.cfg, logger=self.log,
                                 push_queue=push_queue)

        self.tournament = TournamentSelection(config=self.cfg)

        self.macro_mutation = MacroMutations(config=self.cfg,
                                             replay_sample_queue=sample_queue)

        self.micro_mutation = MicroMutations(config=self.cfg,
                                             replay_sample_queue=sample_queue)

        self.training = TD3Training(config=self.cfg,
                                    replay_sample_queue=sample_queue)

        self.individual_micro_counter = 1

    def get_and_increment_ind_micro_counter(self):
        val = self.individual_micro_counter
        self.individual_micro_counter += 1
        return val

    def update_cell_fitnesses(self, population: List[IndividualMacro]) -> None:
        assert(len(population) > 0)
        assert(isinstance(population[0], IndividualMacro))

        for ind in population:
            ind.update_cell_fitnesses()

    # TODO: determine whether or not fitness is copied
    def copy_individuals_with_cell(self, individuals_with_cell:
                                   Dict[int, List[IndividualMacro]])\
            -> Dict[int, List[IndividualMacro]]:
        assert(len(individuals_with_cell) > 0)

        ind_w_cell_copy = {}
        for cid in individuals_with_cell:
            ind_w_cell_copy[cid] = []

        for cid in individuals_with_cell:
            for ind in individuals_with_cell[cid]:
                ind_w_cell_copy[cid].append(ind.clone())
        return ind_w_cell_copy

    def get_cell_active_population(self, population: List[IndividualMacro])\
            -> Set[int]:
        assert(len(population) > 0)

        cell_active_population = set()
        for ind in population:
            cell_active_population.union(ind.get_active_population())
        return cell_active_population

    def get_cell_count_in_macro_individual(self, cell_id: int, macro_individual: IndividualMacro):
        count = 0
        count += macro_individual.actor.get_cell_count()
        count += macro_individual.critic_1.get_cell_count()
        count += macro_individual.critic_1.get_cell_count()

        return count

    def get_individuals_with_cell(self, macro_population:
                                  List[IndividualMacro],
                                  cell_ids: List[int]) -> Dict[int,
                                                               List[IndividualMacro]]:
        assert(len(macro_population) > 0)

        individuals_with_cell = {}
        for cid in cell_ids:
            individuals_with_cell[cid] = []

        for cid in cell_ids:
            for ind in macro_population:
                if ind.contains_cell(cid):
                    individuals_with_cell[cid].append(ind)
        return individuals_with_cell

    def copy_ind_micro_cell_class_without_cells(self,
                                                micro_population:\
                                                Dict[int, IndividualMicro])\
                                                -> Dict[int, IndividualMicro]:
        new_cells = []
        for individual_micro_id in micro_population:
            individual_micro = micro_population[individual_micro_id]
            new_cells[self.individual_micro_counter] = individual_micro.\
                clone_without_cell_copies(self.individual_micro_counter)
            self.individual_micro_counter += 1
        return new_cells

    def merge_new_cells_copied_individuals(
            self, new_cells: List[IndividualMicro],
            copied_individuals: Dict[int, List[IndividualMicro]]) -> None:
        # note cell here is the micro class, not underlying
        # for all cells
        for cell in new_cells:
            cid = cell.id
            # for individuals in that cell
            for ind in copied_individuals[cid]:
                # List[EvolvableMLPCell]
                cells_in_ind = ind.get_cells(cid)
                # get cells of that type, and add to the new cell
                for ev_MLPCell in cells_in_ind:
                    cell.cell_copies_in_population.append(ev_MLPCell)

    #
    def get_micro_individual_from_cell_ids(self, micro_population: Dict[int, IndividualMicro],
                                cell_ids: List[int]) -> Dict[int, IndividualMicro]:
        cell_ids = set(cell_ids)
        #in_cell_ids = []
        #for individual_micro_id in micro_population:
        #    if individual_micro_id in cell_ids:
        #        in_cell_ids[individual_micro_id] = micro_population[individual_micro_id]
        micro_individuals = []
        for cell_id in cell_ids:
            temp_micro_ind = micro_population[cell_id]
            micro_individuals.append(temp_micro_ind)

        return micro_individuals

    def update_cell_mean_fitness(self, micro_population: 
            Dict[int, IndividualMicro]) -> None:
        assert(len(micro_population.keys()) > 0)

        for individual_micro_id in micro_population:
            individual_micro = micro_population[individual_micro_id]
            if individual_micro.active_population:
                cell_fitnesses = [cell.fitness for cell in \
                    individual_micro.cell_copies_in_population]
                individual_micro.set_mean_fitness(sum(cell_fitnesses)\
                    / len(cell_fitnesses))

    def initial_population_micro(self) -> Dict[int, IndividualMicro]:
        self.log.log("initialize micro population")
        population = {}
        for idx in range(self.cfg.nevo.cell_pop_size):
            if self.cfg.nevo.init_random_micro:
                cell_config = copy.deepcopy(self.cfg.cell.get_dict)
                cell_config["activation"] = np.random.choice(
                    ['relu', 'tanh', 'elu'], 1)[0]
                self.log(f"init {idx} cell_config: ", cell_config)
            else:
                cell_config = copy.deepcopy(self.cfg.cell.get_dict)

            # TODO: figure out what to initialize cell with
            cell = EvolvableMLPCell(id=idx, num_inputs=self.cfg.state_dim,
                                    num_outputs=self.cfg.action_dim,
                                    **cell_config)
            individual_micro = IndividualMicro(id=idx, cell=cell)
            population[idx] = individual_micro
            self.individual_micro_counter += 1
        return population

    def initial_population_macro(self, micro_population) -> List[IndividualMacro]:
        self.log.log("initialize macro population")
        population = []
        for idx in range(self.cfg.nevo.population_size):

            if self.cfg.nevo.ind_memory:
                replay_memory = ReplayMemory(
                    capacity=self.cfg.train.replay_memory_size,
                    batch_size=self.cfg.rl.batch_size)
            else:
                replay_memory = False

            if self.cfg.nevo.init_random:

                min_lr = 0.00001
                max_lr = 0.005

                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.rl)

                actor_config["activation"] = np.random.choice(
                    ['relu', 'tanh', 'elu'], 1)[0]
                critic_config["activation"] = np.random.choice(
                    ['relu', 'tanh', 'elu'], 1)[0]

                lr_actor = np.exp(np.random.uniform(
                    np.log(min_lr), np.log(max_lr), 1))[0]
                lr_critic = np.exp(np.random.uniform(
                    np.log(min_lr), np.log(max_lr), 1))[0]

                rl_config.set_attr("lr_actor", lr_actor)
                rl_config.set_attr("lr_critic", lr_critic)
                self.log(f"init {idx} rl_config: ", rl_config.get_dict)
                self.log(f"init {idx} actor_config: ", actor_config)

            else:
                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.rl)


            #input settings to default to random initialization
            indi = IndividualMacro(
                rand_init=True, micro_population = micro_population,
                state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                actor_config=actor_config, critic_config=critic_config,
                rl_config=rl_config, index=idx,
                td3_double_q=self.cfg.train.td3_double_q,
                replay_memory=replay_memory)
            population.append(indi)
        return population


    def add_to_micro_population(self, ind_micro: IndividualMicro, micro_population: dict):
        assert(not ind_micro.id in micro_population)
        micro_population[ind_micro.id] = ind_micro
        return micro_population

    def close(self):
        self.replay_memory.close()

    def evolve_hierarchical_SEARL(
            self, micro_population: Dict[int, IndividualMicro],
            macro_population: List[IndividualMacro],
            n_cells_mutate: int, num_frames:int = 0, epoch:int = 0):

        num_frames = num_frames
        # potential issues:

        # if you mutate cells
        # you have a copy of individuals with each cell (individuals doubled)
        # but you may not mutate all of those individuals

        # by tracking the ancestry of a cell,
        # you could formulate a better way to

        # increased complexity
        # one solution is to just perform less mutations
        # parallel programming of evaluating several mutations in parallel

        # in order to evaluate cells or individuals,
        # you evaluate individuals, but you need to track
        # which cells are connected to which individuals
        # otherwise, this is essentially the same as
        # evaluate was previously
        # EVALUATE SAME BUT CELL TRACKING
        # and then with the cell tracking you need to return cell fitness
        # values for each cell for selection

        # SELECT should be very similar,
        # but you need to be able to apply it to
        # cells also

        # mean fitness is appended to the end of fitness
        # in evaluate

        # then population[i].fitness[-1] used for mean fitness of each individual to select

        # for fitness values in individuals
        # for all cells in individual
        # cell.fitness.append(individual_fitness)

        # for all cells, take mean of fitness to be new fitness

        # maybe append to end of fitness,
        # then you can run through tournament selection if that
        # is what we end up using

        # it might be nice to have a list of active cell population
        # or references to

        # why might you want to select cells differently than individuals?
        # harder to measure cell fitness, and more indirect
        # because cell fitness is so stochastic

        # you can write a function update cell fitness
        # after evaluation is run

        # do you actually want to prune the population of cells much?
        # depends on how cells are selected
        # maybe write a pruning function with options to
        # prune in different ways

        frames_since_mut = 0
        num_frames = num_frames
        epoch = epoch

        # figure out what this does exactly
        ctx = mp.get_context('spawn')

        while True:
            # select subset of micro population??
            pool = ctx.Pool(processes=self.cfg.nevo.worker,
                            maxtasksperchild=1000)

            for ind in macro_population:
                ind.train_log['epoch'] = epoch

            # these may still need to be rewritten because functions
            # may not generalize, but should stay the same

            # TODO FIX MULTIPROCESSING!
            # EVALUATION
            population_mean_fitness, population_var_fitness, eval_frames = \
                self.log.log_func(self.eval.evaluate_population, population=macro_population,
                                  exploration_noise=self.cfg.eval.exploration_noise,
                                  total_frames=num_frames, pool=pool)

            # UPDATE CELL FITNESSES
            self.update_cell_fitnesses(macro_population)
            # Update mean cell fitnesses

            # for all cells in population,
            # take mean fitness value if in active population
            self.update_cell_mean_fitness(micro_population)

            # increment and log
            num_frames += eval_frames
            frames_since_mut += eval_frames
            self.log.population_info(
                population_mean_fitness, population_var_fitness, macro_population,
                num_frames, epoch)

            # save populations
            """
            self.ckp.save_object(
                macro_population, name="macro_population")
            self.ckp.save_object(micro_population, name="micro_population")
            self.log.log("save population")
            """
            if not self.cfg.nevo.ind_memory:
                rm_dict = self.replay_memory.save()
                if isinstance(rm_dict, str):
                    self.log("save replay memory failed")
                else:
                    self.log("replay memory size", len(rm_dict['memory']))
                self.ckp.save_object([rm_dict], name="replay_memory")
                self.log("save replay memory")

            if num_frames >= self.cfg.train.num_frames:
                break


            # TOURNAMENT SELECTION
            if self.cfg.nevo.selection:
                elite, macro_population = self.log.log_func(
                    self.tournament.select_ind_macro,
                    macro_population, micro_population)
                test_fitness = self.eval.test_individual(elite, epoch)
                self.log(f"##### ELITE INFO {epoch}", time_step=num_frames)
                self.log("best_test_fitness", test_fitness, num_frames)


            # TODO add mutations
            # parameter for mutation percentage
            # MUTATION:...
            #   MICRO MUTATION
            if self.cfg.nevo.micro_mutation:

                # for simplicity, copy entire macro_population.
                # This can be changed to copying macro_individuals
                # that have been affected by cell mutation.
                #macro_population_copy = macro_population.clone()

                # set of cell ids (integers)
                cell_active_population = self.get_cell_active_population(
                    macro_population)
                n_cells = len(cell_active_population)
                cell_active_population_arr = np.array(
                    list(cell_active_population))

                # choose cells for mutation with replacement
                # note we could add a config for with/without replacement
                # list of cell ids
                cells_for_mutation = np.random.choice(
                    cell_active_population_arr,
                    size=min(n_cells, n_cells_mutate),
                    replace=True).tolist()

                # get individuals with cell
                # for each different individualMicro, you have a list of all the individualMacros that use that cell
                # dictionary mapping cell_id to list of individualMacros
                individuals_with_cell = self.get_individuals_with_cell(
                    macro_population, cells_for_mutation)

                # get cell classes
                # List[IndividualMicro]. With replacement.
                cell_pop_for_mutation = self.get_micro_individual_from_cell_ids(
                    micro_population, cells_for_mutation)

                #FOR ALL MICRO INDIVIDUALS, MUTATE (WHICH RETURNS A FULL CLONE), THEN COPY MACRO
                #INDIVIDUALS CONTAINING THAT MICRO INDIVIDUALS CELLS
                new_macro_individuals = []
                for ind_micro in cell_pop_for_mutation:
                    original_id = ind_micro.id
                    #this sets the new ids for the individual micro and contained cells
                    new_ind_micro = self.log.log_func(
                        self.micro_mutation.mutation, [ind_micro], self)

                    cells_in_pop = new_ind_micro.cell_copies_in_population.keys().copy()

                    macro_individuals_with_id = individuals_with_cell[original_id]

                    for macro_ind in macro_individuals_with_id:
                        #obtain cell count in macro individual
                        cell_count = macro_ind.actor.get_cell_count()
                        actor_cells = [cells_in_pop.pop() for _ in range(macro_ind.actor.get_cell_count())]
                        critic_1_cells = [cells_in_pop.pop() for _ in range(macro_ind.critic_1.get_cell_count())]
                        critic_2_cells = [cells_in_pop.pop() for _ in range(macro_ind.critic_2.get_cell_count())]

                        new_macro_ind = macro_ind.clone_and_insert_mutated_cells(cell_id_to_change=original_id,
                            actor_mutated_cells=actor_cells, critic_1_mutated_cells=critic_1_cells,
                            critic_2_mutated_cells=critic_2_cells, micro_ind_population_dict=micro_population)

                        new_macro_individuals.append(new_macro_ind)

                    #add mutated micro individual to micro population
                    self.add_to_micro_population(new_ind_micro, micro_population)

                #merge new macro individuals with macro population


                # in here you could incorporate some limit for how many cells get mutated in an individual
                # but then the new cells need to be added back to the original cell class instead of the
                # copied one. otherwise they won't be referenced.

                #basically this assigns the copied evolvable cells to the 
                #corresponding cell class
                
                # TODO: potential issues: does the old class have references to these cells too?
                # TODO: you need to update the cell ids at some point

                # individual micro 1 + individual micro 3 were mutated
                # all macro individuals with cells from ind mic 1 or 3 have both those cells mutated
                
                # non-mutated macro individuals + fully-mutated macro individuals
                # non-mutated macro individuals + macro individuals with ONE micro-mutation

                # construct overall population with non-mutated and mutated macro individuals
                if self.cfg.micro_mutation.keep_original_population:
                    macro_population = macro_population + new_macro_individuals
                else:
                    assert(False)
                    macro_population = new_macro_individuals

            #   MACRO MUTATION
            if self.cfg.nevo.macro_mutation:
                # TODO get the entire population including copies from micro mutations in here
                if self.cfg.nevo.micro_mutation:
                    macro_population = self.log.log_func(
                        self.macro_mutation.mutation, macro_population,
                        micro_population)
                else:
                    macro_population = self.log.log_func(
                        self.macro_mutation.mutation, macro_population)
                # TODO figure out if we really need to do this
                # need to clone in order to update linear layers dimensions
                macro_population = macro_population.clone()


            #TRAINING
            if self.cfg.nevo.training:
                macro_population = self.log.log_func(
                    self.training.train, population=macro_population,
                    eval_frames=eval_frames, pool=pool)

def start_searl_micromacro_run(config, expt_dir):
    with Supporter(experiments_dir=expt_dir, config_dict=config,
                   count_expt=True) as sup:
        cfg = sup.get_config()
        log = sup.get_logger()

        env = gym.make(cfg.env.name)
        cfg.set_attr("action_dim", env.action_space.shape[0])
        cfg.set_attr("state_dim", env.observation_space.shape[0])

        searl = SEARLforTD3(config=cfg, logger=log, checkpoint=sup.ckp)

        micro_population = searl.initial_population_micro()
        macro_population = searl.initial_population_macro(micro_population)

        searl.evolve_hierarchical_SEARL(micro_population, macro_population, n_cells_mutate=cfg.micro_mutation.n_cells_mutate)
