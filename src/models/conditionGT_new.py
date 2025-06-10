
import torch
import torch.nn as nn

from .transformer_model import GraphTransformer
from src.models.model.encoder import Encoder
from src.models.model.nmr_encoder import NMR_encoder

class ConditionGT(nn.Module):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(),
                 dim_enc_H, dimff_enc_H,dim_enc_C, dimff_enc_C, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()
        self.GT = GraphTransformer(n_layers=n_layers_GT,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=act_fn_in,
                                      act_fn_out=act_fn_out)

        # self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len, d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers_TE, drop_prob=drop_prob, device=device)
        self.NMR_encoder = NMR_encoder(device=device, dim_H=dim_enc_H, dimff_H=dimff_enc_H, dim_C=dim_enc_C, dimff_C= dimff_enc_C,
                                       hidden_dim=ffn_hidden, n_head=n_head, num_layers=n_layers_TE, drop_prob=drop_prob)

        self.device = device

        checkpoint = torch.load('/home/liuxuwei01/molecular2molecular/pretrained/new_molen_pre_de.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        GT_state_dict = {k[len('model.GT.'):]: v for k, v in state_dict.items() if
                         k.startswith('model.GT.')}
        # 加载到模型的 conditionEn 部分
        self.GT.load_state_dict(GT_state_dict)

        checkpoint = torch.load('/home/liuxuwei01/molecular2molecular/pretrained/new_molen_pre_de_new_nmr.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        NMR_encoder_state_dict = {k[len('model.NMR_encoder.'):]: v for k, v in state_dict.items() if
                              k.startswith('model.NMR_encoder.')}
        # 加载到模型的 conditionEn 部分
        self.NMR_encoder.load_state_dict(NMR_encoder_state_dict)


    def forward(self, X, E, y, node_mask, condition):

        condition_nmr = self.NMR_encoder(condition)   # 2维张量 （batch, dim）

        y = torch.hstack((y, condition_nmr)).float()

        return self.GT(X, E, y, node_mask)
