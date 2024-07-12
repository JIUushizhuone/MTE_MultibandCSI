
# Import
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from datetime import datetime
from DataProcessingModule import positional_encoding


class ViT(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(ViT, self).__init__()
        self.d_model = d_model
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model * 8)  
        self.fc = nn.Linear(d_model * 8, 3) 

    def forward(self, src, src_key_padding_mask):
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.flatten(start_dim=1)  
        output = self.dropout(output)
        # output = self.norm(output)
        output = self.fc(output)
        return output