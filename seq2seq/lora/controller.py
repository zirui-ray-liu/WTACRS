"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LoRALayer
from ..approxlinear import ApproxLinear


class LoRALinearController(nn.Linear, LoRALayer):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        config=None,
        approx_config=None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.tasks = config.tasks
        self.use_single_lora = config.use_single_lora

        r = config.lora_dim
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                            merge_weights=True)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_As = nn.ModuleDict(dict())
        self.lora_Bs = nn.ModuleDict(dict())
        # Actual trainable parameters
        if r > 0:
            self.lora_As, self.lora_Bs = self.construct_lora_weights(self.tasks, approx_config)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.reset_parameters()


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_As'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for task in self.tasks:
                nn.init.kaiming_uniform_(self.lora_As[task].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_Bs[task].weight)

    def forward(self, x, task=None):
        def T(w):
            return w.T if self.fan_in_fan_out else w
            
        result = F.linear(x, T(self.weight), bias=self.bias)
        
        if task is None:
            task = self.tasks[0]
        
        lora_A = self.lora_As[task]
        lora_B = self.lora_Bs[task]

        if self.training:
            x = self.lora_dropout(x)
            x = lora_A(x)
            x = lora_B(x)
            result += x * self.scaling
        else:
            x = lora_A(x)
            x = lora_B(x)
            result += x * self.scaling

        return result

    def get_task(self, task):
        return task 

    def construct_lora_weights(self, tasks, approx_config):
        if self.use_single_lora:
            if approx_config is not None:
                lora_A = ApproxLinear(self.in_features, self.r, bias=False, config=approx_config)
                # lora_B = ApproxLinear(self.r, self.out_features, bias=False, config=approx_config)
                lora_B = torch.nn.Linear(self.r, self.out_features, bias=False)
            else:
                lora_A = torch.nn.Linear(self.in_features, self.r, bias=False)
                lora_B = torch.nn.Linear(self.r, self.out_features, bias=False)
            for task in tasks:
                self.lora_As[task] = lora_A
                self.lora_Bs[task] = lora_B
        else:
            for task in tasks:
                if approx_config is not None:
                    lora_A = ApproxLinear(self.in_features, self.r, bias=False, config=approx_config)
                    # lora_B = ApproxLinear(self.r, self.out_features, bias=False, config=approx_config)
                    lora_B = torch.nn.Linear(self.r, self.out_features, bias=False)
                else:
                    lora_A = torch.nn.Linear(self.in_features, self.r, bias=False)
                    lora_B = torch.nn.Linear(self.r, self.out_features, bias=False)
                self.lora_As[task] = lora_A
                self.lora_Bs[task] = lora_B

        return self.lora_As, self.lora_Bs

# class LoRALinearController(nn.Linear, LoRALayer):
#     """Implements Adapter controller module which controls the logics of
#     putting adapter layers within transformer's layers."""

#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         config=None,
#         approx_config=None,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)

#         self.tasks = config.tasks
#         self.use_single_lora = config.use_single_lora

#         r = config.lora_dim
#         lora_alpha = config.lora_alpha
#         lora_dropout = config.lora_dropout

#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                             merge_weights=True)

#         self.fan_in_fan_out = fan_in_fan_out
#         self.lora_As = nn.ParameterDict(dict())
#         self.lora_Bs = nn.ParameterDict(dict())
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_As, self.lora_Bs = self.construct_lora_weights(self.tasks)
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.T

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_As'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             for task in self.tasks:
#                 nn.init.kaiming_uniform_(self.lora_As[task], a=math.sqrt(5))
#                 nn.init.zeros_(self.lora_Bs[task])

#     def forward(self, x, task):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w

#         result = F.linear(x, T(self.weight), bias=self.bias)

#         lora_A = self.lora_As[task]
#         lora_B = self.lora_Bs[task]

#         if self.training:
#             result += (self.lora_dropout(x) @ lora_A.T @ lora_B.T) * self.scaling
#         else:
#             result += (x @ lora_A.T @ lora_B.T) * self.scaling

#         return result

#     def get_task(self, task):
#         return task 

#     def construct_lora_weights(self, tasks):
#         if self.use_single_lora:
#             lora_A = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
#             lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))
#             for task in tasks:
#                 self.lora_As[task] = lora_A
#                 self.lora_Bs[task] = lora_B
#         else:
#             for task in tasks:
#                 self.lora_As[task] = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
#                 self.lora_Bs[task] = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))

#         return self.lora_As, self.lora_Bs

