import torch
import torch.nn as nn

from .mpn import MPN, MPNDiff
from ..features.featurization import ATOM_FDIM, BOND_FDIM
from .base_model import FFN


class ReactionModel(nn.Module):
    def __init__(self,
                 mpnn_hidden_size: int = 300,
                 mpnn_bias: bool = True,
                 mpnn_depth: int = 3,
                 mpnn_dropout=0.2,
                 mpnn_diff_hidden_size: int = 300,
                 mpnn_diff_bias: bool = True,
                 mpnn_diff_depth: int = 3,
                 mpnn_diff_dropout=0.2,
                 ffn_hidden_size: int = 300,
                 ffn_bias: bool = True,
                 ffn_dropout=0.2,
                 ffn_depth: int = 3,
                 task_num: int = 2,
                 task_type: str = 'no_softplus'):
        super(ReactionModel, self).__init__()
        self.encoder = MPN(bond_fdim=ATOM_FDIM + BOND_FDIM,
                           atom_fdim=ATOM_FDIM,
                           MPN_hidden_size=mpnn_hidden_size,
                           MPN_bias=mpnn_bias,
                           MPN_depth=mpnn_depth,
                           MPN_dropout=mpnn_dropout,
                           return_atom_hiddens=True)
        self.diff_encoder = MPNDiff(atom_fdim=mpnn_hidden_size,
                                    bond_fdim=ATOM_FDIM + BOND_FDIM,
                                    MPNDiff_hidden_size=mpnn_diff_hidden_size,
                                    MPNDiff_bias=mpnn_diff_bias,
                                    MPNDiff_depth=mpnn_diff_depth,
                                    MPNDiff_dropout=mpnn_diff_dropout)
        self.ffn = FFN(reacvec_fdim=mpnn_diff_hidden_size,
                       ffn_hidden_size=ffn_hidden_size,
                       ffn_dropout=ffn_dropout,
                       ffn_num_layers=ffn_depth,
                       task_num=task_num,
                       ffn_bias=ffn_bias,
                       task_type=task_type)

    def forward(self,
                r_inputs: list,
                p1_inputs: list,
                p2_inputs: list,
                gpu: int):
        r_atom_features = self.encoder.forward(r_inputs, gpu=gpu)
        p1_atom_features = self.encoder.forward(p1_inputs, gpu=gpu)
        p2_atom_features = self.encoder.forward(p2_inputs, gpu=gpu)

        diff_features1 = p1_atom_features - r_atom_features
        diff_features2 = p2_atom_features - r_atom_features
        # for two reactions, we must ensure the results of input r1, r2 equal to that of r1, r1
        # therefore ,features should be summed together
        reaction_features = self.diff_encoder(diff_features1+diff_features2, r_inputs, gpu=gpu)
        output = self.ffn(reaction_features)

        return output


def build_model(hidden_size: int = 300,
                mpnn_depth: int = 3,
                mpnn_diff_depth: int = 3,
                ffn_depth: int = 3,
                use_bias: bool = True,
                dropout=0.2,
                task_num: int = 2,
                ffn_last_layer: str = 'no_softplus'):
    """
    This function is to build model for ranking reactions.
    We minimize the varibles by constrain all hidden_size, dropout, and bias to be the same one.
    For reactions, the param 'return_atom_hiddens' should always be true.
    param: ffn_last_layer: 'with_softplus' or 'no_softplus'
    """
    if task_num == 2 and ffn_last_layer != 'evidential':
        task_type = 'gaussian_' + ffn_last_layer
    elif task_num == 4:
        task_type = 'evidential_' + ffn_last_layer
    elif task_num == 1:
        task_type = ffn_last_layer
    else:
        task_type = ffn_last_layer
    model = ReactionModel(mpnn_hidden_size=hidden_size,
                          mpnn_bias=use_bias,
                          mpnn_depth=mpnn_depth,
                          mpnn_dropout=dropout,
                          mpnn_diff_hidden_size=hidden_size,
                          mpnn_diff_bias=use_bias,
                          mpnn_diff_depth=mpnn_diff_depth,
                          mpnn_diff_dropout=dropout,
                          ffn_hidden_size=hidden_size,
                          ffn_bias=use_bias,
                          ffn_dropout=dropout,
                          ffn_depth=ffn_depth,
                          task_num=task_num,
                          task_type=task_type)

    return model

