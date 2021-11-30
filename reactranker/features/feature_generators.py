from .featurization import str_to_mol
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import Callable, List, Union
from rdkit.Chem import MACCSkeys


from typing import Callable, List, Union

Molecule = Union[str, Chem.Mol]

def morgan_binary_features_generator(mol: Molecule,
                                     radius: int,
                                     num_bits: int) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1-D numpy array containing the binary Morgan fingerprint.
    """
    mol = str_to_mol(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features

def morgan_counts_features_generator(mol: Molecule,
                                     radius: int,
                                     num_bits: int) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.=
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = str_to_mol(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features

def MACCS_features_generator(mol: Molecule) -> np.ndarray:
    """
    Generates MACCS keys fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    """
    mol = str_to_mol(mol) if type(mol) == str else mol
    features_vec = MACCSkeys.GenMACCSKeys(mol)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features
    
def feature_generate(name: str,
                     smiles: List[str],
                     radius: int = 2,
                     num_bits: int = 2048):
    """
    generate additional features
    
    name: the type of the generated features
    including: binary_morgan_fingerprint, counts_based_morgan_fingerprint, MACCS_keys_fingerprint
    smiles: a batch of smiles
    if use Morgan fingerprint, following parameters should be given
    radius: Morgan fingerprint radius
    num_bits: Number of bits in Morgan fingerprint
    """
    features_batch = []
    if name == "binary_morgan_fingerprint":
        for smile in smiles:
            features = morgan_binary_features_generator(mol = smile, radius = radius, num_bits = num_bits)
            features_batch.append(features)
    
    elif name == "counts_based_morgan_fingerprint":
        for smile in smiles:
            features = morgan_counts_features_generator(mol = smile, radius = radius, num_bits = num_bits)
            features_batch.append(features)
            
    elif name == "MACCS_keys_fingerprint":
        for smile in smiles:
            features = MACCS_features_generator(mol = smile)
            features_batch.append(features)
            
    else:
        print("Error:  the name is unknow")
        
    return features_batch