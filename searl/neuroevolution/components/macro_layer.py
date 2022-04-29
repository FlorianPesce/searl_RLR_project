from typing import List

import torch
import torch.nn as nn

from searl.neuroevolution.components.cell import EvolvableMLPCell

# should we implement MacroLayer as a Module with a ModuleList cells attribute,
# or should we implement MacroLayer as a ModuleList directly?
# issue is I am not sure the Cell parameters will be recognized as parameters of
# EvolvableMacroNetwork. Should be an easy change anyway. 

class MacroLayer(nn.Module):
    def __init__(self, cells: List[EvolvableMLPCell]):
        super(MacroLayer, self).__init__()
        self.cells = nn.ModuleList(cells)

    def count_id_in_macrolayer(self, id: int) -> int:
        count = 0
        for cell in self.cells:
            if cell.id == id:
                count += 1
        return count

    def get_output_dims(self):
        return [cell.num_outputs for cell in self.cells]

    def get_input_dims(self):
        return [cell.num_inputs for cell in self.cells]

    def split_tensor(self, x: torch.Tensor, dims_list: List[int])\
            -> List[torch.Tensor]:
        return list(torch.split(x, split_size_or_sections=dims_list, dim = -1))

    def forward(self, x: torch.Tensor):
        input_dims = self.get_input_dims()
        x = self.split_tensor(x=x, dims_list=input_dims)
        for i in range(len(x)):
            if not isinstance(x[i], torch.Tensor):
                x[i] = torch.FloatTensor(x[i])
            x[i] = self.cells[i](x[i])

        # new addition to combine the tensor
        return torch.cat(x, dim=-1)

    def add_cell(self, cell: EvolvableMLPCell) -> None:
        self.cells.append(cell)

    def clone(self, micro_ind_population_dict):
        new_cells = []
        for cell in self.cells:
            #go into micro ind pop
            #find cell class
            micro_ind = micro_ind_population_dict[cell.id]
            new_cell = cell.clone(None)
            #find last id
            micro_ind.add_cell(new_cell)
            assert(new_cell.id != None)
            new_cells.append(cell.clone())
        
        return MacroLayer(new_cells)

    def clone_with_mutations(self, micro_ind_population_dict: dict,
            mutated_cells_to_add: List[EvolvableMLPCell], cell_id_to_change: int):
        
        new_cells = []
        for cell in self.cells:
            #go into micro ind pop
            #find cell class
            if cell.id == cell_id_to_change:
                new_cell = mutated_cells_to_add.pop()
                
                #transfer params
                new_net = new_cell.create_net()
                #transfer params from old to new
                new_net = new_cell.preserve_parameters(old_net = cell.net, 
                        new_net=new_net)
                new_cell.net = new_net
                
                #append new cell
                new_cells.append(new_cell)
            else:
                micro_ind = micro_ind_population_dict[cell.id]
                new_cell = cell.clone(None)
                #find last id
                micro_ind.add_cell(new_cell)
                assert(new_cell.id != None)
                new_cells.append(cell.clone())
        
        assert(len(mutated_cells_to_add) == 0)
        return MacroLayer(new_cells)
        


    """
    def forward(self, x: List[torch.Tensor]):
        output_list = []*len(x)
        for i in range(len(x)):
            if not isinstance(x[i], torch.Tensor):
                x[i] = torch.FloatTensor(x[i])
                output_list[i] = self.cells[i](x[i])

        #new addition to combine the tensor
        return torch.cat(output_list, dim = -1)
    """
       