import copy
import time

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
from searl.neuroevolution.components.cell import EvolvableMLPCell
from searl.neuroevolution.components.individual_td3_macro import IndividualMacro
from searl.neuroevolution.components.replay_memory import (MPReplayMemory,
                                                           ReplayMemory)
from searl.neuroevolution.evaluation_td3 import MPEvaluation
from searl.neuroevolution.mutation_macro import MacroMutations
from searl.neuroevolution.mutation_micro import MicroMutations
from searl.neuroevolution.tournament_selection import TournamentSelection
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
            ["epoch", "time_string", "eval_eps", "pre_fitness", "pre_rank", "post_fitness", "post_rank", "index",
             "parent_index", "mutation", "train_iterations",
             ] + list(self.cfg.rl.get_dict.keys()))

        self.log.log("initialize replay memory")
        if self.cfg.nevo.ind_memory:
            push_queue = None
            sample_queue = None
        else:
            self.replay_memory = MPReplayMemory(seed=self.cfg.seed.replay_memory,
                                                capacity=self.cfg.train.replay_memory_size,
                                                batch_size=self.cfg.rl.batch_size,
                                                reuse_batch=self.cfg.nevo.reuse_batch)
            push_queue = self.replay_memory.get_push_queue()
            sample_queue = self.replay_memory.get_sample_queue()

        self.eval = MPEvaluation(config=self.cfg, logger=self.log, push_queue=push_queue)

        self.tournament = TournamentSelection(config=self.cfg)

        self.macro_mutation = MacroMutations(config=self.cfg, replay_sample_queue=sample_queue)

        self.micro_mutation = MicroMutations(config=self.cfg, replay_sample_queue=sample_queue)

        self.training = TD3Training(config=self.cfg, replay_sample_queue=sample_queue)

    def update_cell_fitnesses(self, population):
        assert(len(population) > 0)
        assert(isinstance(population[0], IndividualMacro))

        for ind in population:
            ind.update_cell_fitnesses()

    def update_cell_mean_fitness(self, cell_population):
        assert(len(cell_population > 0))
        assert(isinstance(cell_population[0], EvolvableMLPCell))

        for cell in cell_population:
            if cell.active_population:
                cell.mean_fitness = sum(cell.fitness) / len(cell.fitness)



    def initial_population_micro(self):
        self.log.log("initialize micro population")
        population = []
        for idx in range(self.cfg.nevo.cell_pop_size):
            if self.cfg.nevo.init_random_micro:
                cell_config = copy.deepcopy(self.cfg.cell.get_dict)
                cell_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]
                self.log(f"init {idx} actor_config: ", cell_config)
            else:
                cell_config = copy.deepcopy(self.cfg.cell.get_dict)
            
            # TODO: figure out what to initialize cell with
            cell = EvolvableMLPCell()
            population.append(cell)
        return population

    def initial_population_macro(self):
        self.log.log("initialize macro population")
        population = []
        for idx in range(self.cfg.nevo.population_size):

            if self.cfg.nevo.ind_memory:
                replay_memory = ReplayMemory(capacity=self.cfg.train.replay_memory_size,
                                             batch_size=self.cfg.rl.batch_size)
            else:
                replay_memory = False

            if self.cfg.nevo.init_random:

                min_lr = 0.00001
                max_lr = 0.005

                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.rl)

                actor_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]
                critic_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]

                lr_actor = np.exp(np.random.uniform(np.log(min_lr), np.log(max_lr), 1))[0]
                lr_critic = np.exp(np.random.uniform(np.log(min_lr), np.log(max_lr), 1))[0]

                rl_config.set_attr("lr_actor", lr_actor)
                rl_config.set_attr("lr_critic", lr_critic)
                self.log(f"init {idx} rl_config: ", rl_config.get_dict)
                self.log(f"init {idx} actor_config: ", actor_config)

            else:
                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.rl)

            indi = Individual(state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                              actor_config=actor_config,
                              critic_config=critic_config,
                              rl_config=rl_config, index=idx, td3_double_q=self.cfg.train.td3_double_q,
                              replay_memory=replay_memory)
            population.append(indi)
        return population

    # TODO
    def evolve_population_micro():
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=self.cfg.nevo.worker, maxtasksperchild=1000)
        
        for ind in population:
            ind.train_log['epoch'] = epoch

        # problems with selecting from the original
        # population, could also be that you
        # don't give mutations a chance to grow, and

        # how do you evaluate a population of cells?
        # you need to get the corresponding population of individuals, grouped by cell, and then
        # evaluate the individuals, then return the fitness function for each cell
        # you could do this one cell at a time
        # for cell, find individuals, evaluate each, save fitness
        # for now, I guess you can just find

        # you could iterate through all the individuals, and find the corresponding ones with that cell
        # or

        #what order?

        #evaluate population

        #log population info

        #save population

        #tournament selection

        #mutation

        #training

        pool.terminate()
        pool.join()

        #select some cells for mutation in population
        population_subset = population#random subset


        self.macro_mutation.mutation()



        


    def evolve_population_macro(self, population, epoch=1, num_frames=0):

        frames_since_mut = 0
        num_frames = num_frames
        epoch = epoch
        ctx = mp.get_context('spawn')

        while True:
            pool = ctx.Pool(processes=self.cfg.nevo.worker, maxtasksperchild=1000)
            epoch_time = time.time()
            self.log(f"##### START EPOCH {epoch}", time_step=num_frames)

            for ind in population:
                ind.train_log['epoch'] = epoch

            population_mean_fitness, population_var_fitness, eval_frames = \
                self.log.log_func(self.eval.evaluate_population, population=population,
                                  exploration_noise=self.cfg.eval.exploration_noise,
                                  total_frames=num_frames, pool=pool)
            num_frames += eval_frames
            frames_since_mut += eval_frames

            self.log.population_info(population_mean_fitness, population_var_fitness, population, num_frames, epoch)

            self.ckp.save_object(population, name="population")
            self.log.log("save population")
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

            if self.cfg.nevo.selection:
                elite, population = self.log.log_func(self.tournament.select, population)
                test_fitness = self.eval.test_individual(elite, epoch)
                self.log(f"##### ELITE INFO {epoch}", time_step=num_frames)
                self.log("best_test_fitness", test_fitness, num_frames)

            if self.cfg.nevo.mutation:
                population = self.log.log_func(self.mutation.mutation, population)

            if self.cfg.nevo.training:
                population = self.log.log_func(self.training.train, population=population, eval_frames=eval_frames,
                                               pool=pool)

            self.log(f"##### END EPOCH {epoch} - runtime {time.time() - epoch_time:6.1f}", time_step=num_frames)
            self.log("epoch", epoch, time_step=num_frames)
            self.log(f"##### ################################################# #####")
            self.cfg.expt.set_attr("epoch", epoch)
            self.cfg.expt.set_attr("num_frames", num_frames)
            epoch += 1

            pool.terminate()
            pool.join()

        self.log("FINISH", time_step=num_frames)
        self.replay_memory.close()

    def close(self):
        self.replay_memory.close()

    def evolve_hierarchical_SEARL(self, micro_population, macro_population):

        #potential issues:

        #if you mutate cells
        #you have a copy of individuals with each cell (individuals doubled)
        #but you may not mutate all of those individuals

        #by tracking the ancestry of a cell,
        #you could formulate a better way to

        #increased complexity
            #one solution is to just perform less mutations
        #parallel programming of evaluating several mutations in parallel

        #in order to evaluate cells or individuals,
        #you evaluate individuals, but you need to track
        #which cells are connected to which individuals
        #otherwise, this is essentially the same as
        #evaluate was previously
        #EVALUATE SAME BUT CELL TRACKING
        #and then with the cell tracking you need to return cell fitness
        #values for each cell for selection

        #SELECT should be very similar,
        #but you need to be able to apply it to
        #cells also

        #mean fitness is appended to the end of fitness
        #in evaluate

        #then population[i].fitness[-1] used for mean fitness of each individual to select

        #for fitness values in individuals
            #for all cells in individual
                #cell.fitness.append(individual_fitness)

        #for all cells, take mean of fitness to be new fitness

        #maybe append to end of fitness,
        #then you can run through tournament selection if that
        #is what we end up using

        # it might be nice to have a list of active cell population
        # or references to

        # why might you want to select cells differently than individuals?
        # harder to measure cell fitness, and more indirect
        # because cell fitness is so stochastic

        #you can write a function update cell fitness
        #after evaluation is run

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
            #select subset of micro population??

            for ind in individual_population:
                ind.train_log['epoch'] = epoch

            #these may still need to be rewritten because functions
            #may not generalize, but should stay the same

            # TODO FIX MULTIPROCESSING!
            #EVALUATION
            population_mean_fitness, population_var_fitness, eval_frames = \
                self.log.log_func(self.eval.evaluate_population, population=population,
                                  exploration_noise=self.cfg.eval.exploration_noise,
                                  total_frames=num_frames, pool=pool)

            #UPDATE CELL FITNESSES
            self.update_cell_fitnesses(individual_population)
            # Update mean cell fitnesses

            # for all cells in population,
            # take mean fitness value if in active population
            self.update_cell_mean_fitness(cell_population)

            # increment and log
            num_frames += eval_frames
            frames_since_mut += eval_frames
            self.log.population_info(population_mean_fitness, population_var_fitness, population, num_frames, epoch)

            #save populations
            self.ckp.save_object(individual_population, name="individual_population")
            self.ckp.save_object(cell_population, name="cell_population")
            self.log.log("save population")
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
                elite, population = self.log.log_func(self.tournament.select_ind, population)
                test_fitness = self.eval.test_individual(elite, epoch)
                self.log(f"##### ELITE INFO {epoch}", time_step=num_frames)
                self.log("best_test_fitness", test_fitness, num_frames)


            #parameter for mutation percentage
            #MUTATION:...
            #   MICRO MUTATION
            #   MACRO MUTATION

            #

            #initialize population



def start_searl_td3_run(config, expt_dir):
    with Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True) as sup:
        cfg = sup.get_config()
        log = sup.get_logger()

        env = gym.make(cfg.env.name)
        cfg.set_attr("action_dim", env.action_space.shape[0])
        cfg.set_attr("state_dim", env.observation_space.shape[0])

        searl = SEARLforTD3(config=cfg, logger=log, checkpoint=sup.ckp)

        macro_population = searl.initial_population_macro()
        searl.evolve_population_macro(macro_population)
