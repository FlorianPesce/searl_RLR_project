from typing import List

import numpy as np
import torch
import torch.nn as nn

from searl.neuroevolution.components.cell import EvolvableMLPCell


class MacroLayer(nn.Module):
    def __init__(self, cells: List[EvolvableMLPCell]):
        super(MacroLayer, self).__init__()

        self.cells = cells

    def get_output_dims(self):
        return [cell.num_outputs for cell in self.cells]

    def get_input_dims(self):
        return [cell.num_inputs for cell in self.cells]

    def forward(self, x: List[torch.Tensor]):
        output_list = []*len(x)
        for i in range(len(x)):
            if not isinstance(x[i], torch.Tensor):
                x[i] = torch.FloatTensor(x[i])
                output_list[i] = self.cells[i](x[i])

        return output_list

        




    


        