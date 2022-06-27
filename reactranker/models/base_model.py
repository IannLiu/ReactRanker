import torch
import torch.nn as nn

from .mpn import MPN, MPNDiff
from ..features.featurization import ATOM_FDIM, BOND_FDIM


class FFN(nn.Module):

    def __init__(self,
                 reacvec_fdim: int,
                 ffn_hidden_size: int,
                 ffn_dropout: float = 0.2,
                 ffn_num_layers: int = 3,
                 task_num: int = 2,
                 ffn_bias: bool = True,
                 task_type: str = 'gaussian',
                 ):

        super(FFN, self).__init__()
        self.hidden_size = reacvec_fdim
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout = ffn_dropout
        self.ffn_num_layers = ffn_num_layers
        self.activation = nn.ReLU()
        self.task_type = task_type
        self.bias = ffn_bias
        self.output = None

        dropout = nn.Dropout(self.dropout)
        activation = self.activation
        if self.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(self.hidden_size, task_num, bias=self.bias)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=self.bias)
            ]
            for _ in range(self.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size, bias=self.bias)
                    # nn.BatchNorm1d(self.ffn_hidden_size, affine=False)
                ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.ffn_hidden_size, task_num, bias=self.bias)
            ])
        self.ffn = nn.Sequential(*ffn)

    def forward(self, x):
        output = self.ffn(x).squeeze(-1)
        if self.task_type == 'evidential_with_softplus':
            min_val = 1e-6
            # Split the outputs into the four distribution parameters
            mu, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1] // 4, dim=1)
            lambdas = torch.nn.Softplus()(loglambdas) + min_val
            alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
            betas = torch.nn.Softplus()(logbetas) + min_val

            # Return these parameters as the output of the model
            self.output = torch.stack((mu, lambdas, alphas, betas), dim=2).view(output.size())
        elif self.task_type == 'gauss_regression_with_softplus':
            # Split the outputs into the four distribution parameters
            mu, logvariance = torch.split(output, output.shape[1] // 2, dim=1)
            variance = torch.nn.Softplus()(logvariance)
            self.output = torch.stack((mu, variance), dim=2).view(output.size())
            # print(self.output)
        elif self.task_type == 'gaussian_with_softplus':
            # Split the outputs into the four distribution parameters
            mu, logvariance = torch.split(output, output.shape[1] // 2, dim=1)
            variance = torch.nn.Softplus()(logvariance)
            self.output = torch.stack((mu, variance), dim=2).view(output.size())
            # print(self.output)
        elif self.task_type == 'listnetdis_lognorm_with_softplus':
            min_val = 1e-6
            # Split the outputs into the four distribution parameters
            mu, logvariance = torch.split(output, output.shape[1] // 2, dim=1)
            mu = torch.nn.Softplus()(mu) + min_val
            variance = torch.nn.Softplus()(logvariance) + min_val
            self.output = torch.stack((mu, variance), dim=2).view(output.size())
            print(self.output)
        elif self.task_type == 'evidential_ranking':
            min_val = 1e-6
            score, uncertain_factor = torch.split(output, output.shape[1] // 2, dim=1)
            # print(output)
            # score = torch.nn.Softplus()(score)
            # uncertain_factor = torch.nn.Sigmoid()(uncertain_factor)
            # uncertain_factor = torch.nn.Softplus()(uncertain_factor) + min_val
            uncertain_score = torch.nn.Softplus()(uncertain_factor)
            self.output = torch.stack((score, uncertain_score), dim=2).view(output.size())
        elif self.task_type == 'listnet_with_softplus':
            self.output = torch.nn.Softplus()(output)
        elif self.task_type == 'listnet_with_uncertainty':
            self.output = torch.nn.Softplus()(output) + 1
        elif self.task_type == 'evidential':
            self.output = torch.nn.Softplus()(output) + 1
        else:
            self.output = output

        return self.output


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
                p_inputs: list,
                gpu: int):
        r_atom_features = self.encoder.forward(r_inputs, gpu=gpu)
        p_atom_features = self.encoder.forward(p_inputs, gpu=gpu)
        """
        if feature_batch is not None:#This choice can not be used because of inputs is not smiles
            diff_features = p_atom_features - r_atom_features
            rfeatures_MACCS = torch.Tensor(feature_generate(name = 'MACCS_keys_fingerprint', smiles=rsmiles)).float()
            pfeatures_MACCS = torch.Tensor(feature_generate(name = 'MACCS_keys_fingerprint', smiles=psmiles)).float()
            react_MACCS = torch.cat([rfeatures_MACCS, pfeatures_MACCS], dim = 1)
            diff_MACCS = pfeatures_MACCS - rfeatures_MACCS
        
        if feature_batch is not None:
            output = self.ffn(self.diff_encoder(diff_features, p_inputs, gpu = gpu, features_batch =react_MACCS ))
        """
        diff_features = p_atom_features - r_atom_features
        output = self.ffn(self.diff_encoder(diff_features, p_inputs, gpu=gpu))

        return output


class ReactionModel_bimol(nn.Module):
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
        super(ReactionModel_bimol, self).__init__()
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
        self.query_encoder = MPN(bond_fdim=ATOM_FDIM + BOND_FDIM,
                                 atom_fdim=ATOM_FDIM,
                                 MPN_hidden_size=mpnn_hidden_size,
                                 MPN_bias=mpnn_bias,
                                 MPN_depth=mpnn_depth,
                                 MPN_dropout=mpnn_dropout)
        self.ffn = FFN(reacvec_fdim=mpnn_diff_hidden_size,
                       ffn_hidden_size=ffn_hidden_size,
                       ffn_dropout=ffn_dropout,
                       ffn_num_layers=ffn_depth,
                       task_num=task_num,
                       ffn_bias=ffn_bias,
                       task_type=task_type)

    def forward(self,
                r_inputs: list,
                p_inputs: list,
                q_inputs: list,
                gpu: int):
        """
        q_inputs
        """
        r_atom_features = self.encoder.forward(r_inputs, gpu=gpu)
        p_atom_features = self.encoder.forward(p_inputs, gpu=gpu)
        q_mol_features = self.query_encoder.forward(q_inputs, gpu=gpu)
        diff_features = p_atom_features - r_atom_features
        output = self.ffn(torch.cat([self.diff_encoder(diff_features, p_inputs, gpu=gpu), q_mol_features]), dim=1)

        return output


def build_model(hidden_size: int = 300,
                mpnn_depth: int = 3,
                mpnn_diff_depth: int = 3,
                ffn_depth: int = 3,
                use_bias: bool = True,
                dropout=0.2,
                task_num: int = 2,
                ffn_last_layer: str = 'no_softplus',
                task_type=None,
                bimolecule=False):
    """
    This function is to build model for ranking reactions.
    We minimize the varibles by constrain all hidden_size, dropout, and bias to be the same one.
    For reactions, the param 'return_atom_hiddens' should always be true.
    param: ffn_last_layer: 'with_softplus' or 'no_softplus'
    """
    if task_type is None:
        if task_num == 2:
            task_type = 'gaussian_' + ffn_last_layer
        elif task_num == 4:
            task_type = 'evidential_' + ffn_last_layer
        elif task_num == 1:
            task_type = ffn_last_layer
        else:
            task_type = ffn_last_layer
    elif task_type == 'evidential_ranking':
        task_type = task_type
    else:
        task_type = task_type + '_' + ffn_last_layer
    if bimolecule is False:
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
    else:
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
