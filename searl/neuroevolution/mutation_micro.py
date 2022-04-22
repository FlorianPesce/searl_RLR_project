import fastrand
import numpy as np

class MicroMutations():
    def __init__(self, config, replay_sample_queue):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.mutation)
        self.replay_sample_queue = replay_sample_queue

    # TODO
    def no_mutation_layer(self, layer):
        individual.train_log["mutation"] = "no_mutation"
        return individual 

    def no_mutation_cell(self, individual_micro):
        individual_micro.train_log["mutation"] = "no_mutation"
        return individual_micro

    # TODO
    def architecture_mutate(self, individual):

        offspring_actor = individual.actor.clone()
        offspring_critic_1 = individual.critic_1.clone()
        if self.cfg.train.td3_double_q:
            offspring_critic_2 = individual.critic_2.clone()

        rand_numb = self.rng.uniform(0, 1)
        if rand_numb < self.cfg.mutation.new_layer_prob:
            offspring_actor.add_layer()
            offspring_critic_1.add_layer()
            if self.cfg.train.td3_double_q:
                offspring_critic_2.add_layer()
            individual.train_log["mutation"] = "architecture_new_layer"
        else:
            node_dict = offspring_actor.add_node()
            offspring_critic_1.add_node(**node_dict)
            if self.cfg.train.td3_double_q:
                offspring_critic_2.add_node(**node_dict)
            individual.train_log["mutation"] = "architecture_new_node"

        individual.actor = offspring_actor
        individual.critic_1 = offspring_critic_1
        if self.cfg.train.td3_double_q:
            individual.critic_2 = offspring_critic_2
        return individual
