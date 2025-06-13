import torch
import torch.nn as nn
from src.diffusion.extra_features import ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from .transformer_model import GraphTransformer
# from .transformer_c_model import GraphTransformer_C
from src.models.model.encoder import Encoder
from src.models.model.nmr_encoder import NMR_encoder
from .molecular_encoder import MolecularEncoder
from src import utils
class DatasetInfo:
    def __init__(self):
        self.valencies = None
        self.atom_weights = None
        self.max_n_nodes = None
        self.max_weight = None
        self.remove_h = None
class contrastGT(nn.Module):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), dim_enc_H, dimff_enc_H, dim_enc_C, dimff_enc_C,
                                 ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()

        # self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len, d_model=d_model, ffn_hidden=ffn_hidden,
        #                        n_head=n_head, n_layers=n_layers_TE, drop_prob=drop_prob, device=device)
        # self.linear_layer = nn.Linear(max_len * d_model, 512)
        self.NMR_encoder = NMR_encoder(device=device, dim_H=dim_enc_H, dimff_H=dimff_enc_H, dim_C=dim_enc_C,
                                       dimff_C=dimff_enc_C,
                                       hidden_dim=ffn_hidden, n_head=n_head, num_layers=n_layers_TE,
                                       drop_prob=drop_prob)

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

        # self.con_input_dim = input_dims
        # self.con_input_dim['X'] = input_dims['X'] - 8
        # self.con_input_dim['y'] = 1024
        # self.con_output_dim = output_dims
        # # self.con_output_dim['y'] = 1024
        # self.conditionEn = GraphTransformer_C(n_layers=n_layers_GT,
        #                               input_dims=self.con_input_dim,
        #                               hidden_mlp_dims=hidden_mlp_dims,
        #                               hidden_dims=hidden_dims,
        #                               output_dims=self.con_output_dim,
        #                               act_fn_in=act_fn_in,
        #                               act_fn_out=act_fn_out)

        self.device = device

        checkpoint = torch.load('/home/liuxuwei01/molecular2molecular/src/new_molen_pre_de.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        conditionEn_state_dict = {k[len('model.conditionEn.'):]: v for k, v in state_dict.items() if
                                  k.startswith('model.conditionEn.')}
        # # 加载到模型的 conditionEn 部分
        self.conditionEn.load_state_dict(conditionEn_state_dict)

        # checkpoint = torch.load('/home/liuxuwei01/PaddleMaterial/output/DeNMR_CHnmr/CLIP/torch-step2-best.ckpt')
        # state_dict = checkpoint['state_dict']
        # conditionEn_state_dict = {k[len('model.conditionEn.'):]: v for k, v in state_dict.items() if
        #                           k.startswith('model.conditionEn.')}
        # self.conditionEn.load_state_dict(conditionEn_state_dict)
        # nmr_encoder_state_dict = {k[len('model.NMR_encoder.'):]: v for k, v in state_dict.items() if
        #                           k.startswith('model.NMR_encoder.')}
        # self.NMR_encoder.load_state_dict(nmr_encoder_state_dict)

        for param in self.conditionEn.parameters():
            param.requires_grad = False

    # def make_src_mask(self, src):
    #     src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     return src_mask

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

    def forward(self, X, E, y, node_mask, X_condition, E_condition, conditionVec):

        # assert isinstance(conditionVec, torch.Tensor), "conditionVec should be a tensor, but got type {}".format(
        #     type(conditionVec))
        #
        # srcMask = self.make_src_mask(conditionVec).to(self.device)
        # conditionVecNmr = self.transEn(conditionVec, srcMask)
        # conditionVecNmr = conditionVecNmr.view(conditionVecNmr.size(0), -1)
        # conditionVecNmr = self.linear_layer(conditionVecNmr)
        conditionVecNmr = self.NMR_encoder(conditionVec)

        # y_condition = torch.zeros(X_condition.size(0), 1024).cuda()
        # conditionVecM = self.conditionEn(X_condition, E_condtion, y_condition, node_mask)

        y_condition = torch.zeros((X.size(0), 0), dtype=torch.float)
        processed_data = self.preprocess_molecular_data(X_condition, E_condition, y_condition, node_mask)
        extra_data = self.compute_extra_data(processed_data)
        X_condition = torch.cat((processed_data['X_t'], extra_data.X), dim=2).float()
        E_condition = torch.cat((processed_data['E_t'], extra_data.E), dim=3).float()
        y_condition = torch.hstack((processed_data['y_t'], extra_data.y)).float()
        # import numpy as np
        # np.save("/home/liuxuwei01/PaddleMaterial/output/X_condition.npy", X_condition.detach().cpu().numpy())
        # np.save("/home/liuxuwei01/PaddleMaterial/output/E_condition.npy", E_condition.detach().cpu().numpy())
        # np.save("/home/liuxuwei01/PaddleMaterial/output/y_condition.npy", y_condition.detach().cpu().numpy())
        # np.save("/home/liuxuwei01/PaddleMaterial/output/node_mask.npy", node_mask.detach().cpu().numpy())
        conditionVecM = self.conditionEn(X_condition, E_condition, y_condition, node_mask)

        return conditionVecM,conditionVecNmr