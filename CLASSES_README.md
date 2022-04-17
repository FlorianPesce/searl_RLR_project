Notes:

individuals each have their own replay memory
(if this is collective, unsure why each individual has one?)
SEARL makes it sound like this is collective and shared between
all agents

our cells do not have an individual outer class
cells need a fitness list which is contained in individual outer class
cells also need a clone method which is contained in the individual class
I guess 

TODO:
    someone needs to write clone for evolvable 




class Individual 

self.actor = EvolvableNetwork() 

self.critic_1 = EvolvableNetwork() 

  

class EvolvableNetwork 

self.layers = [layer1, layer2, layer2, etc.] 

  

class Layer 

self.cells = [cell1, cell1, cell2, etc.] 

 

  

Their setup: 

Class Individual # definitions for td3
Class EnvolvableMLP
Create network 

Forward
Add layer (copies network and then adds parameters 

 

 

 

 

class Cell
self.num_inputs = 200
self.num_outputs = 50
Def add_node
Def add_layer 

  

  

class IndividualMutations 

def no_mutation
def rl_hyperparam_mutation
def add_layer 

  

class CellMutations 

def no_mutation
def architecture_mutate 

  

class LayerMutations 

def no_mutation
def activate_mutation
def _permutate_activation
def add_cell 