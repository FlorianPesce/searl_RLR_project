issues:

TODO

CREATE GLOBAL CELL ID COUNT: NEW CELLS INITIALIZED TO COUNT + 1 AND COUNT INCREMENTED

MAKE SURE CELL IDS ARE UPDATED AFTER WHATEVER COPY FUNCTION IS USED

MUTATIONS USE DIFFERENT COPYING PARADIGMS
	COPY INDIVIDUAL MICRO CLASS IN CHANGE ACTIVATIONS MUTATION

FIX: CLONE IN EVOLVABLE MACRO NETWORK DOESN'T COPY THE LAYERS AND THE CELLS, 

	AND WHEN THIS IS DONE, DO THE NEW CELLS GET ASSIGNED TO THE ORIGINAL CLASS?
	PRESUMABLY THEY SHOULD
	
	WE SHOULD ALSO UPDATE THE MACRO LAYERS (DO WE COPY THEM OR INSERT THE NEW CELL)


WRITE A FUNCTION WHICH EITHER 
	
	1. CLONES INDIVIDUALS AND SIMULTANEOSLY INSERTS MUTATED CELLS
	
	or
	
	2. INSERTS MUTATED CELLS
	
	TO BE USED FOR THE FOLLOWING MUTATION METHOD:
	MOST CONSISTENT WITH FLORIANS CODE 
	copy micro class and cells first, mutate, copy macro individuals and insert cells (either during ind copy, or after) update individual	


#how to purge dead cells (within an individual micro class)?
	
	#give pytorch evolvable cells intra class id
	#(sufficient to remove dead cells)
	
	#alternatively add dictionary of cells in individual micro
	#and make cell population a dictionary

    #create count















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