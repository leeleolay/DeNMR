import torch
import torch.nn as nn
from src.diffusion.extra_features import ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from .transformer_model import GraphTransformer
# from .transformer_c_model import GraphTransformer_C
from .molecular_encoder import MolecularEncoder
from src import utils

class DatasetInfo:
    def __init__(self):
        self.valencies = None
        self.atom_weights = None
        self.max_n_nodes = None
        self.max_weight = None
        self.remove_h = None
class molecularGT(nn.Module):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.GT = GraphTransformer(n_layers=n_layers_GT,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=act_fn_in,
                                      act_fn_out=act_fn_out)

        self.con_input_dim = input_dims
        self.con_input_dim['y'] = 12
        self.con_output_dim = output_dims
        # self.con_output_dim['y'] = 1024
        self.conditionEn = MolecularEncoder(n_layers=n_layers_GT,
                                      input_dims=self.con_input_dim,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=self.con_output_dim,
                                      act_fn_in=act_fn_in,
                                      act_fn_out=act_fn_out)

        self.dataset_infos = DatasetInfo()  # 创建 DatasetInfo 实例
        self.dataset_infos.valencies = [4, 3, 2, 1, 3, 2, 1, 1, 1]  # 设置 valencies
        self.dataset_infos.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 30.97, 5: 32.07, 6: 35.45, 7: 79.9, 8: 126.9}
        self.dataset_infos.max_n_nodes = 15
        self.dataset_infos.max_weight = 564
        self.dataset_infos.remove_h = True

        self.extra_features = ExtraFeatures('all', dataset_info=self.dataset_infos)
        self.domain_features = ExtraMolecularFeatures(dataset_infos=self.dataset_infos)

    def preprocess_molecular_data(self, X, E, y, node_mask):

        z_t = utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)

        data = {'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return data

    def compute_extra_data(self, data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(data)
        extra_molecular_features = self.domain_features(data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def forward(self, X, E, y, node_mask, X_condition, E_condition):
        # y_condition = torch.zeros(X.size(0), 1024).cuda()
        y_condition = torch.zeros((X.size(0), 0), dtype=torch.float)
        # print(f'Xshape{X.shape}')
        # print(f'node_mask{node_mask.shape}')
        # print(f'X_condition{X_condition.shape}')
        processed_data = self.preprocess_molecular_data(X_condition, E_condition, y_condition, node_mask)

        extra_data = self.compute_extra_data(processed_data)
        X_condition = torch.cat((processed_data['X_t'], extra_data.X), dim=2).float()
        E_condition = torch.cat((processed_data['E_t'], extra_data.E), dim=3).float()
        y_condition = torch.hstack((processed_data['y_t'], extra_data.y)).float()
        import numpy as np
        np.save('/home/liuxuwei01/PaddleMaterial/output/X_condition.npy',X_condition.detach().cpu().numpy())
        np.save('/home/liuxuwei01/PaddleMaterial/output/E_condition.npy',E_condition.detach().cpu().numpy())
        np.save('/home/liuxuwei01/PaddleMaterial/output/y_condition.npy',y_condition.detach().cpu().numpy())
        
        conditionVec = self.conditionEn(X_condition, E_condition, y_condition, node_mask)

        y = torch.hstack((y, conditionVec)).float()

        return self.GT(X, E, y, node_mask)