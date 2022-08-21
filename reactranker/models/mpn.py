from typing import List, Union
import numpy as np

import torch
import torch.nn as nn

from ..features.featurization import mol2graph
from ..utils import index_select_ND


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, 
                 bond_fdim: int,
                 atom_fdim: int,
                 MPN_hidden_size: int,
                 MPN_bias: bool = True,
                 MPN_depth: int = 6,
                 MPN_dropout: float = 0.2,
                 return_atom_hiddens: bool = False):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param return_atom_hiddens: Return hidden atom feature vectors instead of mol vector.
        """
        super(MPN, self).__init__()
        self.return_atom_hiddens = return_atom_hiddens
        self.bond_fdim = bond_fdim
        self.atom_fdim = atom_fdim
        self.hidden_size = MPN_hidden_size
        self.bias = MPN_bias
        self.depth = MPN_depth
        self.dropout = MPN_dropout
        self.layers_per_message = 1

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = nn.ReLU()

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)


        w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        if self.depth > 1:
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph,
                gpu: int,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        # print('the length of smiles is', len(smiles))

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        # print('the length of f_atom is:', f_atoms.size())
        if gpu is not None:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(gpu), f_bonds.cuda(gpu), a2b.cuda(gpu), b2a.cuda(gpu), b2revb.cuda(gpu)

        # Input
        input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            # print('for mpnn, now, the depth is:', depth)

            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)
            # merge input，num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden
        
        # last layer, without reverse bonds
        a2x = a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.return_atom_hiddens:
            return atom_hiddens

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)
                
        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPNDiff(nn.Module):
    """A message passing neural network for encoding of custom (difference) features."""

    def __init__(self,
                 atom_fdim: int,
                 bond_fdim: int,
                 MPNDiff_hidden_size: int,
                 MPNDiff_bias: bool = True,
                 MPNDiff_depth: int = 3,
                 MPNDiff_dropout: float = 0.2):
        """Initializes the MPNDiffEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        """
        super(MPNDiff, self).__init__()
        self.atom_fdim = atom_fdim #equal to hidden size
        self.bond_fdim = bond_fdim
        self.hidden_size = MPNDiff_hidden_size
        self.bias = MPNDiff_bias
        self.depth = MPNDiff_depth
        self.dropout = MPNDiff_dropout
        self.layers_per_message = 1

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = nn.ReLU()

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        self.W_i = nn.Linear(self.atom_fdim, self.hidden_size, bias=self.bias)

        # Shared weight matrix across depths (default)
        if self.depth > 1:
            self.W_h = nn.Linear(self.hidden_size + self.bond_fdim, self.hidden_size, bias=self.bias)

        if self.depth > 0:
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                atom_features: torch.FloatTensor,
                mol_graph,
                gpu: int,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs with custom features.

        :param atom_features: Atom features for the BatchMolGraph.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if features_batch is not None:
            features_batch = torch.FloatTensor(features_batch)
            if gpu is not None:
                features_batch = features_batch.cuda(gpu)

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        a2a = mol_graph.get_a2a()
        if gpu is not None:
            f_bonds, a2b, a2a = f_bonds.cuda(gpu), a2b.cuda(gpu), a2a.cuda(gpu)
            
        # Input
        input = self.W_i(atom_features)  # num_atoms x atom_fdim
        message = self.act_func(input)  # num_atoms x hidden_size

        if self.depth > 0:
            # Message passing
            for depth in range(self.depth - 1):
                # print('for mpnn_diff, now, the depth is:', depth)
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x (bond_fdim (+ atom_fdim_MPN))

                # If using bond messages in MPN, bond features include some atom features,
                # but we only want the pure bond features here
                nei_f_bonds = nei_f_bonds[:, :, -self.bond_fdim:]

                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim

                message = self.W_h(message)
                message = self.act_func(input + message)  # num_atoms x hidden_size
                message = self.dropout_layer(message)  # num_atoms x hidden

            nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            a_input = torch.cat([atom_features, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
            atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        else:
            atom_hiddens = self.dropout_layer(message)

        # Readout
        vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                vecs.append(mol_vec)

        vecs = torch.stack(vecs, dim=0)  # (num_samples, hidden_size)
        
        if features_batch is not None:
            vecs = torch.cat([vecs, features_batch], dim=1)

        return vecs  # num_samples x hidden