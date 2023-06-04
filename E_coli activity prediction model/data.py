from rdkit import Chem
import codecs
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from subword_nmt.apply_bpe import BPE
import deepchem as dc
import deepchem.molnet as dcm
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning import LightningDataModule


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smiles2index(s1, words2idx, bpe):
    t1 = bpe.process_line(s1).split()
    i1 = [words2idx[i] for i in t1]
    return i1


def index2multi_hot(i1, idx2word):
    v1 = np.zeros(len(idx2word))
    v1[i1] = 1
    return v1


def index2multi_hot_fg(molecule, fgroups_list):
    v1 = np.zeros(len(fgroups_list))
    for idx in range(len(fgroups_list)):
        if molecule.HasSubstructMatch(fgroups_list[idx]):
            v1[idx] = 1
    return v1


def smiles2vector_fgr(s1, words2idx, bpe, idx2word, fgroups_list):
    i1 = smiles2index(s1, words2idx, bpe)
    mfg = index2multi_hot(i1, idx2word)
    molecule = Chem.MolFromSmiles(s1)
    fg = index2multi_hot_fg(molecule, fgroups_list)
    return fg, mfg


class FsrFgDataset(Dataset):

    def __init__(self, data, idx2word, fgroups_list, words2idx, bpe, descriptor_funcs):
        self.mols = data.X
        self.y = data.y
        self.smiles = data.ids
        self.idx2word = idx2word
        self.fgroups_list = fgroups_list
        self.words2idx = words2idx
        self.bpe = bpe
        self.descriptor_funcs = descriptor_funcs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        mol = self.mols[idx]
        smile = self.smiles[idx]
        target = self.y[idx]

        fg, mfg = smiles2vector_fgr(smile, self.words2idx, self.bpe, self.idx2word, self.fgroups_list)
        num_features = np.asarray([self.descriptor_funcs[key](mol) for key in self.descriptor_funcs.keys()])
        return np.float32(fg), np.float32(mfg), np.float32(num_features), int(target)


class FsrFgDataModule(LightningDataModule):
    def __init__(self, root: str = 'Data', task_name: str = 'ecoli', batch_size: int = 8, num_workers: int = 8,
                 pin_memory: bool = True, split_type: str = 'scaffold', num_folds: int = 5, fold_index: int = 0):
        super().__init__()

        self.root = root
        self.task_name = task_name
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.split_type = split_type
        self.num_folds = num_folds
        self.fold_index = fold_index
        self.test_fold = None
        self.val_fold = None
        self.train_fold = None
        self.fgroups_list = None
        self.descriptor_funcs = None
        self.bpe = None
        self.idx2word = None
        self.words2idx = None
        self.val_data = None
        self.test_data = None
        self.train_data = None
        self.splits = None

    def prepare_data(self) -> None:
        df = pd.read_csv('Data/ecoli_inhibition.csv')
        ids = df['smiles']
        labels = df['Encoded_Activity']
        X = [Chem.MolFromSmiles(smiles) for smiles in ids]
        data = dc.data.DiskDataset.from_numpy(X=X, y=labels, ids=ids)
        fgroups = pd.read_csv(self.root + '/Functional_groups_filtered.csv')
        fgroups_list = list(map(lambda x: Chem.MolFromSmarts(x), fgroups['SMARTS'].tolist()))
        self.fgroups_list = [i for i in fgroups_list if i]

        self.descriptor_funcs = {name: func for name, func in Descriptors.descList}

        vocab_path = self.root + '/codes_drug_chembl_1500.txt'
        bpe_codes_fin = codecs.open(vocab_path)
        self.bpe = BPE(bpe_codes_fin, merges=-1, separator='')
        vocab_map = pd.read_csv(self.root + '/subword_units_map_drug_chembl_1500.csv')
        self.idx2word = vocab_map['index'].values
        self.words2idx = dict(zip(self.idx2word, range(0, len(self.idx2word))))
        self.weights = data.w
        if self.split_type == 'scaffold':
            splitter = dc.splits.ScaffoldSplitter()
            self.splits = []
            for i in range(self.num_folds):
                new_data = data.complete_shuffle()
                self.splits.append((new_data, splitter.split(dataset=new_data, seed=i)))
            self.dataset = FsrFgDataset(self.splits[self.fold_index][0], self.idx2word, self.fgroups_list, self.words2idx, self.bpe,
                                        self.descriptor_funcs)
            self.train_ind, self.val_ind, self.test_ind = self.splits[self.fold_index][1]
        else:
            splitter = dc.splits.RandomStratifiedSplitter()
            self.splits = [splitter.split(dataset=data, seed=fold_num) for fold_num in range(self.num_folds)]
            self.dataset = FsrFgDataset(data, self.idx2word, self.fgroups_list, self.words2idx, self.bpe,
                                        self.descriptor_funcs)
            self.train_ind, self.val_ind, self.test_ind = self.splits[self.fold_index]

    def setup(self, stage=None):
        """define train, test and validation datasets """

        self.train_fold = Subset(self.dataset, self.train_ind)
        self.val_fold = Subset(self.dataset, self.val_ind)
        self.test_fold = Subset(self.dataset, self.test_ind)

    def train_dataloader(self):
        """returns train dataloader"""
        loader = DataLoader(self.train_fold, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)
        return loader

    def val_dataloader(self):
        """returns val dataloader"""
        loader = DataLoader(self.val_fold, batch_size=1, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return loader

    def test_dataloader(self):
        """returns test dataloader"""
        loader = DataLoader(self.test_fold, batch_size=1, shuffle=False,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return loader
