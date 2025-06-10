import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics
# from src.numericalize_text import numericalize_text
# from src.numericalize_1H13C import numericalize_H1C13


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class CHnmrDataset(InMemoryDataset):
    # raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #            'molnet_publish/qm9.zip')
    # raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    # processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, stage, root, remove_h: bool, target_prop=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.target_prop = target_prop
        self.stage = stage
        # self.vocabDim = 256
        self.seq_len_H1 = 20
        self.seq_len_C13 = 75
        '''*********'''
        self.vocab_peakwidth = {"<pad>": 0, "<unk>": 1}
        self.vocab_split = {"<pad>": 0, "<unk>": 1}

        # # 读取 vocab.src 文件并添加到词汇表字典
        # with open("C:/Users/wubl/Desktop/molecular2molecular/src/vocab.src", "r", encoding="utf-8") as vocab_file:
        #     # 从编号 2 开始，因为 0 和 1 分配给了特殊的 token
        #     current_id = 2
        #     for line in vocab_file:
        #         # 每行分割为单词和它的频次
        #         word, _ = line.strip().split("\t")
        #         self.vocab_to_id[word] = current_id
        #         current_id += 1
        peakwidth_filepath = '/home/liuxuwei01/molecular2molecular/data/CHnmr/CHnmr_pyg/statistic/H1_statistic/delta_distribution.csv'
        df_peakwidth =pd.read_csv(peakwidth_filepath)
        for idx, value in enumerate(df_peakwidth['Value']):
            self.vocab_peakwidth[value] = idx + 2

        split_filepath = '/home/liuxuwei01/molecular2molecular/data/CHnmr/CHnmr_pyg/statistic/H1_statistic/split_type_distribution.csv'
        df_split = pd.read_csv(split_filepath)
        for idx, type in enumerate(df_split['Type']):
            self.vocab_split[type] = idx + 2

        '''_________'''

        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])



    @property
    def raw_file_names(self):
        return ['No.sdf', 'tokenized_dataset_N_new.csv', 'uncharacterized.txt']
        # return ['tokenized_dataset_N.csv']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            # file_path = download_url(self.raw_url, self.raw_dir)
            # extract_zip(file_path, self.raw_dir)
            # os.unlink(file_path)
            #
            # file_path = download_url(self.raw_url2, self.raw_dir)
            # os.rename(osp.join(self.raw_dir, '3195404'),
            #           osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = int(0.9 * n_samples)
        n_test = int(0.05 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # 读取 CSV 文件
        '''*******'''
        # target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0, converters={"tokenized_input": json.loads})
        '''-------'''

        data_list = []
        for idx, row in tqdm(target_df.iterrows(), total=target_df.shape[0]):
            smiles = row['smiles']
            tokenized_input = row['tokenized_input']
            atom_count = row['atom_count']

            # 将 SMILES 转化为 mol 对象
            mol = Chem.MolFromSmiles(smiles)
            # if mol is None:
            #     continue

            N = mol.GetNumAtoms()

            # 获取原子类型索引
            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            # 获取边信息
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros((1, 0), dtype=torch.float)

            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]


            # 创建 Data 对象，并包含额外的信息（tokenized_input 和 atom_count）
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=idx)
            '''*******'''
            # data.conditionVec = torch.tensor(numericalize_text(text=tokenized_input, vocab_to_id=self.vocab_to_id, dim=self.vocabDim), dtype=torch.long)
            data.H_nmr, data.num_H_peak, data.C_nmr, data.num_C_peak = numericalize_H1C13(nmrdata=tokenized_input,
                                                                                          vocab_peakwidth=self.vocab_peakwidth,
                                                                                          vocab_split=self.vocab_split,
                                                                                          seq_len_H1=self.seq_len_H1,
                                                                                          seq_len_C13=self.seq_len_C13)
            '''________'''
            data.atom_count = torch.tensor(atom_count, dtype=torch.long)
            # conditionVec = numericalize_text(text = tokenized_input, vocab_to_id=self.vocab_to_id, dim=self.vocabDim)
            # data.tokenized_input = torch.tensor([ord(char) for char in tokenized_input], dtype=torch.long)
            # data.atom_count = torch.tensor(atom_count, dtype=torch.long)
            # data.tokenized_input_length = torch.tensor(len(data.tokenized_input), dtype=torch.long)

            # 过滤和转换
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])  # .pt





class CHnmrDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h

        target = getattr(cfg.general, 'guidance_target', None)
        regressor = getattr(self, 'regressor', None)
        if regressor and target == 'mu':
            transform = SelectMuTransform()
        elif regressor and target == 'homo':
            transform = SelectHOMOTransform()
        elif regressor and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': CHnmrDataset(stage='train', root=root_path, remove_h=cfg.dataset.remove_h,
                                        target_prop=target, transform=RemoveYTransform()),              # .process()
                    'val': CHnmrDataset(stage='val', root=root_path, remove_h=cfg.dataset.remove_h,
                                      target_prop=target, transform=RemoveYTransform()),
                    'test': CHnmrDataset(stage='test', root=root_path, remove_h=cfg.dataset.remove_h,
                                       target_prop=target, transform=transform)}
        super().__init__(cfg, datasets)


class CHnmrinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'CHnmr'
        if self.remove_h:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
            self.atom_decoder = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            self.num_atom_types = 9
            self.valencies = [4, 3, 2, 1, 3, 2, 1, 1, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 30.97, 5: 32.07, 6: 35.45, 7: 79.9, 8: 126.9}
            self.max_n_nodes = 15
            self.max_weight = 564

            self.n_nodes = torch.tensor([0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,6.579831824637949467e-04,3.417267464101314545e-03,9.784846566617488861e-03,1.977487094700336456e-02,4.433957487344741821e-02,7.253380119800567627e-02,1.089563518762588501e-01,1.475509554147720337e-01,1.760564893484115601e-01,1.996448338031768799e-01,2.172830253839492798e-01])
            self.node_types = torch.tensor([7.162184715270996094e-01,9.598348289728164673e-02,1.247809454798698425e-01,1.828921213746070862e-02,4.915347089990973473e-04,1.454589515924453735e-02,1.616295613348484039e-02,1.132413558661937714e-02,2.203370677307248116e-03])
            self.edge_types = torch.tensor([8.293983340263366699e-01,9.064729511737823486e-02,1.195883937180042267e-02,1.138782827183604240e-03,6.685676425695419312e-02])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0: 7] = torch.tensor([0.000000000000000000e+00,1.856458932161331177e-01,2.707855999469757080e-01,3.008204102516174316e-01,2.362315803766250610e-01,3.544347826391458511e-03,2.972166286781430244e-03])

        else:
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
            self.valencies = [1, 4, 3, 2, 1]
            self.num_atom_types = 5
            self.max_n_nodes = 29
            self.max_weight = 390
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            self.n_nodes = torch.tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                         9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                         1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                         1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                         5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

            self.node_types = torch.tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
            self.edge_types = torch.tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)

            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())

            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_CHnmr_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_CHnmr_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting CHnmr dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting CHnmr dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

def numericalize_H1C13(nmrdata, vocab_peakwidth, vocab_split, seq_len_H1, seq_len_C13):
    Hnmr = nmrdata['1HNMR']
    Cnmr = nmrdata['13CNMR']

    num_h1peak = len(Hnmr)
    hnmr_pro = []
    for peak in Hnmr:
        chem_shift = peak[0]
        peakwidth_idx = vocab_peakwidth.get(peak[1], vocab_peakwidth["<unk>"])
        split_idx = vocab_split.get(peak[2], vocab_split["<unk>"])  # 建立模式字典
        integral = int(peak[3].replace("H", "")) + 1

        J_coupling= peak[4]
        padded_J = J_coupling + [0] * (6 - len(J_coupling)) if len(J_coupling) > 0 else [0] * 6

        peak_new = [chem_shift, peakwidth_idx,split_idx,integral] + padded_J
        hnmr_pro.append(peak_new)

    hnmr_tensor = torch.zeros(seq_len_H1, 10, dtype=torch.float32)
    hnmr_tensor[:num_h1peak,:] = torch.tensor(hnmr_pro,dtype=torch.float32)

    num_c13peak = len(Cnmr)
    padded_cnmr = Cnmr + [0] * (seq_len_C13 - num_c13peak)
    cnmr_tensor = torch.tensor(padded_cnmr,dtype=torch.float32)


    return hnmr_tensor, num_h1peak, cnmr_tensor, num_c13peak
