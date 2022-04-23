from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from macro_layer import MacroLayer


#mutate individuals, test individuals select best individuals

#option 1 (all cells)
#mutate cells, test individuals, select best cells based on fitness score

#option 2 (some cells)
#mutate some cells, test individuals, select best cells based on fitness score

#option 3 (one cell)
#test individuals with cell, mutate a cell, test individuals after mutation, obtain performance difference

#***PREFERRED OPTION***#
#option 4 (some cells)
#mutate some cells (keep originals in pop), test individuals, select best cells with tournament selection based on fitness score
# individual has cell1, cell2. Mutation on cell1 and cell2, they become cellN+1 and celln+2.
# Population: cell1, cell2 ; celln+1, cell2 ; cell1, celln+2

#consider speciating cells by size


#fitness score is average of 


#how is input fed to cells?
#fully connected from each input to each cell for now

#how is output computed?
#fully connected from each cell to each output


#how will we store layer transitions
#how will we update layer transitions
class EvolvableMacroNetwork(nn.Module):
    def __init__(self, layers: List[MacroLayer], num_inputs: int, num_outputs: int,
                 activation='relu', output_activation=None) -> None:
        super(EvolvableMacroNetwork, self).__init__()

        self.layers = layers
        self.net = self.create_net()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        self.output_activation = output_activation
        self.contained_active_population = set()

    def update_cell_fitnesses(self, mean_fitness):

        for layer in self.layers:
            for cell in layer.cells:
                cell.fitness.append(mean_fitness)
                cell.active_population = True

    def get_active_population(self):
        active_population = set()
        for layer in self.layers:
            for cell in layer.cells:
                active_population.add(cell.id)
        self.contained_active_population = active_population
        return active_population

    def update_active_population(self):
        for layer in self.layers:
            for cell in layer.cells:
                cell.active_population = True





    #returns an ordered dict of macrolayers and 
    #macro layer connections
    # Todo: in future if we do arbitrary connections between cells, create special linear layer class for this so that we can keep using nn.sequential
    def create_net(self) -> OrderedDict:
        net_dict = OrderedDict()

        #create input linear layer, activation
        layer0_indim = self.layers[0].get_input_dims()
        net_dict[f'linear_layer_input'] = nn.Linear(self.num_inputs, sum(layer0_indim))
        net_dict[f'activation_input'] = self.activation

        #create layer, linear connections 
        for i in range(len(self.layers)-1):
            #add layer to ordered dict and layer connection to ordered dict
            net_dict[f'layer_{i}'] = self.layers[i]
            output_dims = self.layers[i].get_output_dims()
            input_dims = self.layers[i].get_input_dims()
            net_dict[f'linear_layer_{i}'] = nn.Linear(sum(input_dims), sum(output_dims))
            net_dict[f'activation{i}'] = self.activation

        lastlayer_index = len(self.layers) - 1
        lastlayer_outdim = self.layers[-1].get_output_dims()

        #create last layer, and linear for output
        net_dict[f'layer_{lastlayer_index}'] = self.layers[lastlayer_index]
        net_dict[f'linear_layer_output'] = nn.Linear(sum(lastlayer_outdim), self.num_outputs)
        net_dict[f'activation_output'] = self.output_activation

        #return net_dict
        #not sure if this will work with nn.sequential
        return nn.sequential(net_dict)

    #divides tensor into a list of tensors with dimensions
    #specified in dims_list
    def split_tensor(self, x: torch.Tensor, dims_list: List[int]) -> List[torch.Tensor]:
        return list(torch.split(x, split_size_or_sections=dims_list, dim = -1))

    #concatenate list of tensors in first or last dimension
    #return tensor
    def combine_tensor(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, dim = -1)

    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        for layer in self.net:
            #if i is an activation, and not the last activation
            #we know output is passed to a macrolayer, so split tensor
            
            #if macro layer, then 
            if layer.startswith("layer_"):
                x = self.split_tensor(x, self.net[layer].get_input_dims())
                x = self.net[layer](x)
                x = self.combine_tensor(x)
            
            else:
                x = self.net[layer](x)
            
        return x
    """

    def add_layer(self, layer: Optional[MacroLayer] = None):
        if layer:
            # TODO add functionality of copying a layer
            # TODO sample a cell from population of cells and create a layer with it
            self.layers.append(MacroLayer())
        else:
            self.layers.append(layer)

    #add cell to random layer
    def add_cell(self, cell):

    def clone(self):
        clone = EvolvableMacroNetwork(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(self.state_dict())
        return clone


    # Probably easiest to pop a cell off of the end of the network
    # so you don't need to mess with preserve parameters
    #def remove_cell

    #this sets preserved parameters to be at the
    #beginning of new layer
    #so add or remove cells from the end of the list
    #to avoid complications

    #one way to debug this could be to create the module
    #for the linear connector layers. make that a module list
    #of linear layers between cells, so when you call preserve parameters
    #you can copy linear layers directly to linear layers
    def preserve_parameters(self, old_net, new_net):

        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if not "norm" in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        if len(param.data.size()) == 1:
                            param.data[:min(old_size[0], new_size[0])] = old_net_dict[key].data[
                                                                         :min(old_size[0], new_size[0])]
                        else:
                            param.data[:min(old_size[0], new_size[0]),
                                                    :min(old_size[1], new_size[1])] = old_net_dict[
                                                                     key].data[
                                                                 :min(old_size[
                                                                          0],
                                                                      new_size[
                                                                          0]),
                                                                 :min(old_size[
                                                                          1],
                                                                      new_size[
                                                                          1])]

        return new_net

    def shrink_preserve_parameters(self, old_net, new_net):

        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if not "norm" in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[:min_0, :min_1]
        return new_net

