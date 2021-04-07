import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock


class Classifier(nn.Module):
  def __init__(self, d_model=80, n_spks=600, dropout=0.1):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(40, d_model)
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=1024, nhead=1
    )
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.Dropout(p=0.2),
      nn.ReLU(),
      nn.Linear(d_model, n_spks),
    )

    self.block = ConformerBlock(
      dim = 128*80,
      dim_head = 64,
      heads = 8,
      ff_mult = 4,
      conv_expansion_factor = 2,
      conv_kernel_size = 31,
      attn_dropout = 0.,
      ff_dropout = 0.,
      conv_dropout = 0.
  )


  def forward(self, mels):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = self.prenet(mels)
    # out: (length, batch size, d_model)
    # print(out.shape)
    out = out.permute(1, 0, 2)
    # out = out.reshape(out.shape[0], -1)
    # out = self.block(out)

    # The encoder layer expect features in the shape of (length, batch size, d_model).
    out = self.encoder_layer(out)
    # out: (batch size, length, d_model)
    out = out.transpose(0, 1)
    # mean pooling
    stats = out.mean(dim=1)

    # out: (batch, n_spks)
    out = self.pred_layer(stats)
    return out
