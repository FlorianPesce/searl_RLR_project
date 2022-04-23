from typing import List

import numpy as np
import torch
import torch.nn as nn

from searl.neuroevolution.components.cell import EvolvableMLPCell


class MacroLayer(nn.Module):
    def __init__(self, cells: List[EvolvableMLPCell]):
        super(MacroLayer, self).__init__()
        self.cells = nn.ModuleList(cells)

    def get_output_dims(self):
        return [cell.num_outputs for cell in self.cells]

    def get_input_dims(self):
        return [cell.num_inputs for cell in self.cells]

    def split_tensor(self, x: torch.Tensor, dims_list: List[int]) -> List[torch.Tensor]:
        return list(torch.split(x, split_size_or_sections=dims_list, dim = -1))

    def forward(self, x: torch.Tensor):
        input_dims = self.get_input_dims()
        x= self.split_tensor(x=x,dims_list=input_dims)
        for i in range(len(x)):
            if not isinstance(x[i], torch.Tensor):
                x[i] = torch.FloatTensor(x[i])
                output_list[i] = self.cells[i](x[i])

        #new addition to combine the tensor
        return torch.cat(output_list, dim = -1)



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

        




    


        