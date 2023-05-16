import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq.backend.bit_relu_dropout import act_quantized_relu as qrelu


class QReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return qrelu(input)
        else:
            return F.relu(input)