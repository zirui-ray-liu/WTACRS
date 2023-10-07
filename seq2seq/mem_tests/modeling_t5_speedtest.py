import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

import sys
sys.path.append("../../")

from seq2seq.third_party.models.t5.modeling_t5 import T5Attention, T5Block, T5LayerFF, T5ForConditionalGeneration
from seq2seq.third_party.models.t5.configuration_t5 import T5Config


class T5Model(nn.Module):
    
    def __init__(self, config, approx_config=None):
        super().__init__()
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.T5encoder = nn.ModuleList([T5Block(config=encoder_config, has_relative_attention_bias=False, 
                     adapter_config=None, lora_config=None, approx_config=approx_config) for i in range(config.num_layers)]
            )
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.T5decoder = nn.ModuleList([T5Block(config=decoder_config, has_relative_attention_bias=False, 
                     adapter_config=None, lora_config=None, approx_config=approx_config) for i in range(config.num_layers)]
            )

    
    def forward(self, x, **kwargs):
        
        output = x
        for layer in self.T5encoder:
            output = layer(output, **kwargs)
            if isinstance(output, tuple):
                output = output[0]
                
        for layer in self.T5decoder:
            output = layer(output, **kwargs)
            if isinstance(output, tuple):
                output = output[0]
        
        return output