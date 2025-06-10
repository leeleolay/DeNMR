import torch
import torch.nn as nn

from .transformer_model import GraphTransformer
from .transformer_c_model import GraphTransformer_C

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
        self.con_input_dim['X'] = input_dims['X'] - 8
        self.con_input_dim['y'] = 1024
        self.con_output_dim = output_dims
        # self.con_output_dim['y'] = 1024
        self.conditionEn = GraphTransformer_C(n_layers=n_layers_GT,
                                      input_dims=self.con_input_dim,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=self.con_output_dim,
                                      act_fn_in=act_fn_in,
                                      act_fn_out=act_fn_out)
        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/epoch=438.ckpt')

        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']

        # 从 state_dict 中提取 conditionEn 部分的权重
        conditionEn_state_dict = {k[len('model.conditionEn.'):]: v for k, v in state_dict.items() if
                                  k.startswith('model.conditionEn.')}

        # 加载到模型的 conditionEn 部分
        self.conditionEn.load_state_dict(conditionEn_state_dict)

        # 如果成功加载，可以打印一条确认信息
        print("conditionEn parameters loaded successfully.")

        for param in self.conditionEn.parameters():
            param.requires_grad = False

        # self.linear_layer = nn.Linear(225, 512)

    def forward(self, X, E, y, node_mask, X_condition, E_condtion):
        y_condition = torch.zeros(X.size(0), 1024).cuda()
        conditionVec = self.conditionEn(X_condition, E_condtion, y_condition, node_mask)
        # print(f'conditionVec.shape{conditionVec.shape}')
        # conditionVec = conditionVec.X
        # conditionVec = conditionVec.view(X.size(0),-1)
        # conditionVec = self.linear_layer(conditionVec)

        y = torch.hstack((y, conditionVec)).float()

        return self.GT(X, E, y, node_mask), conditionVec