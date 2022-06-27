from rdkit import Chem
from typing import List, Tuple, Union
import torch

RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()


def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:
    
    """
    Converts an InChI or SMILES string to an RDKit molecule.

    :param string: The InChI or SMILES string.
    :param explicit_hydrogens: Whether to treat hydrogens explicitly.
    :return: The RDKit molecule.
    """
    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creat one-hot-vector
    
    value : the value for which element should be one
    choices : a list of one features, it gives the lenthgs of the one-hot vector belonging to the feature
    """

    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1 
    # if value in choices, index = the index of which the values appears, 
    # else, index = -1, which means the last element would be choiced
    encoding[index] = 1
    return encoding


elem_list = ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'I', 'B', 'K']

ATOM_FEATURES = {
    'atomic_symbol' : elem_list,
    'degree' : [0, 1, 2, 3, 4],
    'formal_charge' : [-2, -1, 0, 1, 2],
    'chiral' : [0, 1, 2, 3],
    'num_Hs' : [0, 1, 2, 3, 4],
    'radical': [0, 1, 2, 3, 4],
    'hybridization' : [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
}

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2 + 8
BOND_FDIM = 14 + 8


def atom_features (atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    build atom_features'_ROAtomSeq' object has no attribute 'GetAtomicNum'
    
    param atom : an atom of RDKit object
    param functional groups : A k-hot vector indicating the functional groups the atom belongs to
    return: a list containing the atom features.
    """
    
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atomic_symbol']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(atom.GetChiralTag(), ATOM_FEATURES['chiral']) + \
               onek_encoding_unk(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(atom.GetNumRadicalElectrons(), ATOM_FEATURES['radical']) + \
               onek_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
                                                               
    features += [
        atom.IsInRingSize(3),
        atom.IsInRingSize(4),
        atom.IsInRingSize(5),
        atom.IsInRingSize(6),
        atom.IsInRingSize(7),
        atom.IsInRingSize(8),
        atom.IsInRingSize(9),
        atom.IsInRingSize(10),
    ]
    
    if functional_groups is not None:
        features += functional_groups

    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Bulid a feature bond for a bond
    
    :param bond : a RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM -1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0, #this is a tag
            bt == Chem.BondType.SINGLE,
            bt == Chem.BondType.DOUBLE,
            bt == Chem.BondType.TRIPLE,
            bt == Chem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),  # whether this bond is considered Conjugated
            (bond.IsInRing() if bt is not None else 0),
            (bond.IsInRingSize(3) if bt is not None else 0),
            (bond.IsInRingSize(4) if bt is not None else 0),
            (bond.IsInRingSize(5) if bt is not None else 0),
            (bond.IsInRingSize(6) if bt is not None else 0),
            (bond.IsInRingSize(7) if bt is not None else 0),
            (bond.IsInRingSize(8) if bt is not None else 0),
            (bond.IsInRingSize(9) if bt is not None else 0),
            (bond.IsInRingSize(10) if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

      the following attributes:
    - smiles: Smiles string.<rdkit.Chem.rdchem.Atom object at 0x7fd2b1e33850>
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """
    def __init__(self, smiles: str, reaction: bool = True,
                 atom_messages: bool = False):
        self.smiles = smiles
        self.f_atoms = []
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to cat(atom_features, bond_features)
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the atom index the bond comes from
        self.b2revb = []  # mapping from bond index to the reverse bond index
        
        # convert smiles to molecule
        mol = str_to_mol(smiles, explicit_hydrogens=True)

        self.n_atoms = mol.GetNumAtoms()
        # Require atom numbers when using reactions
        # so that activation atoms can be 
        if reaction:
            # Because of some atom map number in RMG_data is 0.
            # Following code is missed
            # if any(a.GetAtomMapNum() == 0 for a in mol.GetAtoms()):
            #     raise Exception(f'{smiles} is missing atom map numbers')
                
            atoms = sorted(mol.GetAtoms(), key=lambda a: a.GetAtomMapNum())
        else:
            atoms = mol.GetAtoms()
        # Get Atom features
        for i, atom in enumerate(atoms):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
        
        for _ in range(self.n_atoms):
            self.a2b.append([])
        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1+1, self.n_atoms):  # in case of repeating bonds
                rdkit_idx1 = atoms[a1].GetIdx()
                rdkit_idx2 = atoms[a2].GetIdx()
                bond = mol.GetBondBetweenAtoms(rdkit_idx1, rdkit_idx2)
                
                if bond is None:
                    continue
                    
                f_bond = bond_features(bond)
                if atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)
                
                # update index mapping
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1) 
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    
    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """
    def __init__(self, mol_graphs: List[MolGraph], atom_messages: bool = False):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + (not atom_messages) * self.atom_fdim
        # the atom feature and bond featuer. if atom messages is false, which indicate the atom feature should
        # be added together.
        
        # ???
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms) 
            f_bonds.extend(mol_graph.f_bonds)
            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])
                # append the a2b of mol_graph to the a2b of BatchMolGraph, the bond index start at 1
                # every bond of every molecule was added, and the bonds are taken into account
            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
            
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
        
        # Transform
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None
    
    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.
        
        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope
    
    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            print('\n')
            print("b2b is : ", b2b)
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            print("revmask is :", revmask)
            self.b2b = b2b * revmask
        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]
            
        return self.a2a

    def get_smiles(self):
        """
        :return: The smiles of molecules
        """
        return self.smiles_batch

                
def mol2graph(smiles_batch: List[str]) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        mol_graph = MolGraph(smiles)
        mol_graphs.append(mol_graph)
    
    return BatchMolGraph(mol_graphs)
