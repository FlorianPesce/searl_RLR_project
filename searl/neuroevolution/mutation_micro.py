import fastrand
import numpy as np

class MicroMutations():
    def __init__(self, config, replay_sample_queue):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.seed.mutation)
        self.replay_sample_queue = replay_sample_queue

    def no_mutation_layer(self, layer):
        individual.train_log["mutation"] = "no_mutation"
        return individual 


