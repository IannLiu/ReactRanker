import datetime

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from reactranker.features.featurization import BatchMolGraph, MolGraph


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class get_data:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.
    :param path: Path to a CSV file.
    """
    def __init__(self, path):
        self.path = path
        self.num_reactions = None
        self.num_reactants = None
        self.df = None

    def get_num(self):
        """
        :return the number of reactions and reactants
        """
        self.num_reactants = len(self.df.rsmi.unique())
        self.num_reactions = len(self.df)
        print('reaction number is: ', self.num_reactions)
        print('reactant number is: ', self.num_reactants)
        return self.num_reactions, self.num_reactants

    def read_data(self, sep=','):
        print(get_time(), "load file from {}".format(self.path))
        self.df = pd.read_csv(self.path, sep=sep)
        print(get_time(), "finish loading from {}".format(self.path))

    def filter_bacth(self, filter_szie: int = 3):
        """
        Filter the batch which size is lower than filter size

        :param filter_szie: the filtered batch sized
        :return: Flitered data
        """
        df = self.df
        index = []
        for reactant in df.rsmi.unique():
            if df[df.rsmi == reactant].shape[0] < filter_szie:
                index.extend(list(df.index[df.rsmi == reactant]))
        df = df.drop(index=index)

        self.df = df

    def get_all_data(self):
        """
        :return: all loaded data
        """
        return self.df

    @staticmethod
    def shuffle_data(df, seed: int=0):
        """
        Shuffle the data frame

        :param df: The DataFrame to shuffle
        :param seed: The random state
        :return: Shuffled DataFrame
        """
        shuffled_df = df.sample(frac=1, random_state=seed)

        return shuffled_df

    def split_data(self, df=None, split_size=(0.8, 0.1, 0.1), split_type='reactants', seed: int = 0):
        """
        Split the data in terms of reactions or reactants

        :param df: Dataframe to be split
        :param split_size: split size.
        :param split_type: split data according to the reactant number or reaction number.
        :param seed: the random shuffle state
        :return: DataFrame for train, validate, test
        """
        if df is None:
            df = self.df
        if split_type == 'reactions':
            data = df.sample(frac=1, random_state=seed)
            rows, cols = data.shape
            split_index1 = int(rows * split_size[1])
            split_index2 = int(rows * (split_size[2]+split_size[1]))
            test_data = data.iloc[0:split_index1, :]
            val_data = data.iloc[split_index1:split_index2, :]
            train_data = data.iloc[split_index2:rows, :]

            return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)
        elif split_type == 'reactants':
            reactants = df.rsmi.unique()
            reactants_shuffle = shuffle(reactants, random_state=seed)
            rows = len(reactants_shuffle)
            split_index1 = int(rows * split_size[1])
            split_index2 = int(rows * (split_size[2] + split_size[1]))
            reactants_val = reactants_shuffle[0:split_index1]
            reactants_test = reactants_shuffle[split_index1:split_index2]
            reactants_train = reactants_shuffle[split_index2:rows]
            train_data = self.df[self.df.rsmi == reactants_train[0]]
            val_data = self.df[self.df.rsmi == reactants_val[0]]
            test_data = self.df[self.df.rsmi == reactants_test[0]]
            for smi in reactants_train[1:]:
                train_data = pd.concat([train_data, self.df[self.df.rsmi == smi]], axis=0)
            for smi in reactants_val[1:]:
                val_data = pd.concat([val_data, self.df[self.df.rsmi == smi]], axis=0)
            for smi in reactants_test[1:]:
                test_data = pd.concat([test_data, self.df[self.df.rsmi == smi]], axis=0)

            return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)


class DataProcessor:
    """
    Processing all of the needed data, including the data for pairwise and listwise
    """

    def __init__(self, df, num_properties=2):
        """
        :param df: A data frame object
        :param num_properties: The properties
        """
        self.df = df
        self.num_pairs = None
        self.num_reactants = len(df.rsmi.unique())
        self.num_properties = num_properties
        self.smiles2graph = {}

    def get_num_reactants(self):
        return self.num_reactants

    def generate_query_batch(self, df, name='std_targ', batchsize=100000):
        """
        (This function is used for eval_ndcg_at_k)
        Generating batch in spite of query

        :param df: pandas.DataFrame, contains column qid
        :param name: The target name
        :param batchsize: The batch size
        :returns: numpy.ndarray qid, rel, x_i
        """
        if df is None:
            df = self.df
        idx = 0
        while idx * batchsize < df.shape[0]:
            r = df.iloc[idx * batchsize: (idx + 1) * batchsize, :]
            yield r.rsmi.values, r[name].values, r[['rsmi', 'psmi']].values
            idx += 1

    def generate_batch_per_query(self,
                                 df=None,
                                 name='std_targ',
                                 shuffle_query=True,
                                 shuffle_batch=True,
                                 seed=0):
        """
        Get the batch extracted from every query

        :param df: pandas.DataFrame
        :param name: the target property
        :param shuffle_query: Shuffle the query or not
        :param shuffle_batch: Shuffle batch items or not
        :param seed: The random state of shuffle
        :return: X for features, y for relavance
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if df is None:
            df = self.df
        reactants = df.rsmi.unique()
        if shuffle_query:
            reactants = shuffle(reactants, random_state=seed)
        for reactant in reactants:
            df_reactant = df[df.rsmi == reactant]
            if shuffle_batch:
                df_reactant = df_reactant.sample(frac=1, random_state=seed)
            yield df_reactant[['rsmi', 'psmi']].values, df_reactant[name].values

    def generate_batch_querys(self,
                              df=None,
                              batch_size: int = 2,
                              name='std_targ',
                              shuffle_query=True,
                              shuffle_batch=True,
                              seed=0):
        """
        Get the batch extracted from every query

        :param df: pandas.DataFrame
        :param batch_size: The number of queries
        :param name: the target property
        :param shuffle_query: Shuffle the query or not
        :param shuffle_batch: Shuffle batch items or not
        :param seed: The random state of shuffle
        :return: X for features, y for relavance
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if df is None:
            df = self.df
        reactants = df.rsmi.unique()
        if shuffle_query:
            reactants = shuffle(reactants, random_state=seed)
        idx = 0
        smiles, targets, scope = None, None, []
        for reactant in reactants:
            df_reactant = df[df.rsmi == reactant]
            if shuffle_batch:
                df_reactant = df_reactant.sample(frac=1, random_state=seed)
            scope.append(df_reactant.shape[0])
            smile, target = df_reactant[['rsmi', 'psmi']].values, df_reactant[name].values.reshape(-1, 1)
            if smiles is None:
                smiles, targets = smile, target
            else:
                smiles = np.vstack((smiles, smile))
                targets = np.vstack((targets, target))
            idx += 1
            while idx >= batch_size:
                yield smiles, targets, scope
                idx = 0
                smiles, targets, scope = None, None, []
        if idx >= 1:
            yield smiles, targets, scope

    def get_num_pairs(self):
        if self.num_pairs is not None:
            return self.num_pairs
        self.num_pairs = 0
        for _, target in self.generate_batch_per_query(self.df):
            target = target.reshape(-1, 1)
            pairs = target - target.T
            pos_pairs = np.sum(pairs > 0, (0, 1))
            neg_pairs = np.sum(pairs < 0, (0, 1))
            assert pos_pairs == neg_pairs
            self.num_pairs += pos_pairs + neg_pairs
        return self.num_pairs

    def generate_query_pairs(self, df, reactant, targ, seed=0):
        """
        Generating reaction pairs for given reactant

        :param df: pandas.DataFrame, contains column qid, rel, fid from 1 to self.num_features
        :param reactant: reactant smiles
        :param targ: The target name
        :param seed: The random sate for the query items
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        if df is None:
            df = self.df
        df_reactant = df[df.rsmi == reactant]
        if seed is not None:
            df_reactant.sample(frac=1, random_state=seed)
        rels = df_reactant[targ].unique()
        x_i, x_j, y_i, y_j = [], [], [], []
        for r in rels:
            df1 = df_reactant[df_reactant[targ] == r]
            df2 = df_reactant[df_reactant[targ] != r]
            df_merged = pd.merge(df1, df2, on='rsmi')
            df_merged.reindex(np.random.permutation(df_merged.index))
            y_i.append(df_merged[targ + '_x'].values.reshape(-1, 1))
            y_j.append(df_merged[targ + '_y'].values.reshape(-1, 1))
            x_i.append(df_merged[['rsmi', 'psmi_x']].values)
            x_j.append(df_merged[['rsmi', 'psmi_y']].values)
        return np.vstack(x_i), np.vstack(y_i), np.vstack(x_j), np.vstack(y_j)

    def generate_query_pair_batch(self, df=None, targ='std_targ', batchsize=2000, seed=0):
        """
        Generating pair batch for given batch size with queries

        :param df: pandas.DataFrame, contains column qid
        :param targ: The target name
        :param batchsize: Generated batch size. Every batch might have several queries
        :param seed: The random state for queries and batches
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        if df is None:
            df = self.df[['idx', 'rsmi', 'psmi', targ]]
        else:
            df = df[['idx', 'rsmi', 'psmi', targ]]
        x_i_buf, y_i_buf, x_j_buf, y_j_buf = None, None, None, None
        reactants = df.rsmi.unique()
        np.random.seed(seed)
        np.random.shuffle(reactants)
        print(reactants)
        for reactant in reactants:
            x_i, y_i, x_j, y_j = self.generate_query_pairs(df, reactant, targ, seed)
            if x_i_buf is None:
                x_i_buf, y_i_buf, x_j_buf, y_j_buf = x_i, y_i, x_j, y_j
            else:
                x_i_buf = np.vstack((x_i_buf, x_i))
                y_i_buf = np.vstack((y_i_buf, y_i))
                x_j_buf = np.vstack((x_j_buf, x_j))
                y_j_buf = np.vstack((y_j_buf, y_j))
            idx = 0
            while (idx + 1) * batchsize <= x_i_buf.shape[0]:
                start = idx * batchsize
                end = (idx + 1) * batchsize
                yield x_i_buf[start: end, :], y_i_buf[start: end, :], x_j_buf[start: end, :], y_j_buf[start: end, :]
                idx += 1

            x_i_buf = x_i_buf[idx * batchsize:, :]
            y_i_buf = y_i_buf[idx * batchsize:, :]
            x_j_buf = x_j_buf[idx * batchsize:, :]
            y_j_buf = y_j_buf[idx * batchsize:, :]

        yield x_i_buf, y_i_buf, x_j_buf, y_j_buf


class Parsing_features:
    """
    One of the bottle-neck for training reactions is featurization. This process is CPU only.
    This class generates a dictionary to store the generated features. Therefore,  featurization
    can be fastly done.
    """
    def __init__(self):
        self.smiles2graph = {}

    def parsing_smiles(self, smiles: list = None):
        """
        Generating features for smiles list
        :param smiles: Smiles list of molecules
        """
        if smiles is not None:
            mol_graphs = []
            for smi in smiles:
                if smi in self.smiles2graph.keys():
                    mol_graph = self.smiles2graph[smi]
                else:
                    mol_graph = MolGraph(smi, reaction=True, atom_messages=False)
                    self.smiles2graph[smi] = mol_graph
                mol_graphs.append(mol_graph)
            batch_graph = BatchMolGraph(mol_graphs)
            return batch_graph
        else:
            return None

    def parsing_reactions(self, reactions: list = None):
        """
        Generating features for reactions
        :param reactions: A list: [[reactant, product]]
        """
        if reactions is not None:
            rsmi = [s[0] for s in reactions]
            psmi = [s[1] for s in reactions]
            r_batch_graph = self.parsing_smiles(rsmi)
            p_batch_graph = self.parsing_smiles(psmi)
            return [r_batch_graph, p_batch_graph]
        else:
            return [None, None]

    def clear_cache(self):
        """
        clean the cache after finishing all of jobs
        """
        self.smiles2graph.clear()
