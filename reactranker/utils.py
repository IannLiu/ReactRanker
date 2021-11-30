from typing import List
import numpy as np
import csv
import logging
import os

import torch
import torch.nn as nn

from sklearn.utils import shuffle
from .features.featurization import BatchMolGraph, MolGraph
from .data.scaler import StandardScaler



def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def get_data(path: str,
             reaction: bool = True):
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param reaction: Whether loading reactions instead of molecules.
    """
    
    #load_data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # 1skip header

        lines = []
        for line in reader:
            lines.append(line)
            #targets.append(line[3:4])

    return lines

def dealdata (data: list):
    """
    deal data for prediction of reaction properties such as H, Ea, and so on.
    """
    
    rsmiles = [x[1] for x in data]
    psmiles = [x[2] for x in data]
    EH= [x[3:] for x in data]

    scaler = StandardScaler().fit(EH)
    scaled = scaler.transform(EH).tolist()
    targets = torch.Tensor(scaled)
    EH_num = np.array(EH).astype(float)

    return rsmiles, psmiles, scaler, targets, EH_num
    
def dealdata_oc(data):
    """
    deal data for prediction of outcomes.
    """
    len_p = int(data[2])
    psmiles = data[3:(3+len_p)]
    #print('this is psmiles', psmiles)
    target = torch.Tensor(np.array(data[3+len_p:(3+len_p + len_p)]).astype(int))
    #print('this is target', target)
    rsmiles = [data[1]] * int(len_p)
    #print('this is rsmiles', rsmiles)
    
    return rsmiles, psmiles, target
    
def dealdata_pair(data):
    """
    deal data for prediction of outcomes.
    """
    rsmiles1 = [x[1] for x in data]
    psmiles1 = [x[2] for x in data]
    rsmiles2 = [x[3] for x in data]
    psmiles2 = [x[4] for x in data]
    target= torch.Tensor(np.array([x[5] for x in data]).astype(int))
    
    return rsmiles1, psmiles1, rsmiles2, psmiles2, target
    
def dealdata_pair_reduce(data):
    """
    deal data for prediction of outcomes.
    """
    scope = []
    rsmiles = []
    psmiles = []
    targets = []
    for item in data:
        len_p = int(item[2])
        scope.append(len_p)
        rsmiles.extend([item[1]] * int(len_p))
        psmiles.extend(item[3:(3+len_p)])
        targets.extend(list(map(float, item[3+len_p:(3+len_p + len_p)])))
        
    targets = torch.Tensor(targets)
    scope = torch.IntTensor(scope)
    
    return rsmiles, psmiles, targets, scope
    
class dealdata_list:
    """
    deal data for prediction of outcomes.
    param ini_seed: the shuffle state
    """
    def __init__(self,
                 order: bool = True,
                 shuffle_query: bool = True,):

        super(dealdata_list, self).__init__()
        self.order = order
        self.shuffle_query = shuffle_query
        self.smi_to_graph_dict = {}

    def parsing_data(self, data, ini_seed: int = 0):
        scope = []
        rsmiles = []
        psmiles = []
        targets = []
        for item in data:
            len_p = int(item[2])
            scope.append(len_p)
            psmi = item[3:(3+len_p)]
            rsmi = [item[1]] * int(len_p)
            batch_targets = np.array(item[3+len_p:(3+len_p + len_p)]).astype(float)
        
            if not self.order:
                if self.shuffle_query: #shuffle every query
                    index = shuffle(list(range(len_p)), random_state=ini_seed + len_p)
                    shuffled_psmi = [psmi[i] for i in index]
                    shuffled_targets = [batch_targets[i] for i in index]
                    rsmiles.extend(rsmi)
                    psmiles.extend(shuffled_psmi)
                    targets.extend(shuffled_targets)
                else:
                    rsmiles.extend(rsmi)
                    psmiles.extend(psmi)
                    targets.extend(batch_targets)
            else:
                sorted_num = sorted(enumerate(batch_targets), key=lambda x: x[1])
                index = [x[0] for x in sorted_num]
                num = [x[1] for x in sorted_num]
                sorted_rsmi = [rsmi[i] for i in index]
                sorted_psmi = [psmi[i] for i in index]
                rsmiles.extend(sorted_rsmi)
                psmiles.extend(sorted_psmi)
                targets.extend(num)

        r_mol_graphs = []
        p_mol_graphs = []
        for rs, ps in zip(rsmiles, psmiles):
            if rs in self.smi_to_graph_dict.keys():
                r_mol_graph = self.smi_to_graph_dict[rs]
            else:
                r_mol_graph = MolGraph(rs, reaction = True, atom_messages = False)
                self.smi_to_graph_dict[rs] = r_mol_graph
            if ps in self.smi_to_graph_dict.keys():
                p_mol_graph = self.smi_to_graph_dict[ps]
            else:
                p_mol_graph = MolGraph(ps, reaction = True, atom_messages = False)
                self.smi_to_graph_dict[ps] = p_mol_graph
            r_mol_graphs.append(r_mol_graph)
            p_mol_graphs.append(p_mol_graph)

        r_batch_graph = BatchMolGraph(r_mol_graphs)
        p_batch_graph = BatchMolGraph(p_mol_graphs)

        return r_batch_graph, p_batch_graph, torch.Tensor(targets), torch.IntTensor(scope)
    
def dealdata_list_rmg(data, order = True):
    """
    deal data for prediction of outcomes.
    """
    scope = []
    rsmiles = []
    psmiles = []
    targets = []
    
    for item in data:
        len_smi = int(item[0])
        scope.append(len_smi)
        rsmi = item[1:(1+len_smi)]
        psmi = item[(1+len_smi):(1+len_smi*2)]
        batch_targets = np.array(item[(1+len_smi*2):(1+len_smi*3)]).astype(float)
        
        if not order:
            rsmiles.extend(rsmi)
            psmiles.extend(psmi)
            targets.extend(batch_targets)
        else:
            sorted_num = sorted(enumerate(batch_targets), key=lambda x: x[1])
            index = [x[0] for x in sorted_num]
            num = [x[1] for x in sorted_num]
            sorted_rsmi = [rsmi[i] for i in index]
            sorted_psmi = [psmi[i] for i in index]
            rsmiles.extend(sorted_rsmi)
            psmiles.extend(sorted_psmi)
            targets.extend(num)
                      
    return rsmiles, psmiles, torch.Tensor(targets), torch.IntTensor(scope)
        
def save_checkpoint(path:str,
                    model,
                    means= None,
                    stds=None,
                    ):
    """
    Saves a model checkpoint.

    :param model: model
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': means,
            'stds': stds
        } if means is not None and stds is not None else None
    }
    
    torch.save(state, path)
    
    
def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target
        
def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger
    
def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)