import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq.backend.bit_relu_dropout import act_quantized_dropout as qdropout


class QDropout(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return qdropout(input, self.p, self.training)
        else:
            return super(QDropout, self).forward(input)