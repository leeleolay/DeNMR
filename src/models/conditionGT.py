
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_model import GraphTransformer
from src.models.model.encoder import Encoder
from torch.nn.utils.rnn import pack_padded_sequence
class ConditionGT(nn.Module):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()
        self.GT = GraphTransformer(n_layers=n_layers_GT,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=act_fn_in,
                                      act_fn_out=act_fn_out)

        self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len, d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers_TE, drop_prob=drop_prob, device=device)
        # self.max_len = max_len
        # self.d_model = d_model
        # # 定义线性变换层
        # self.query = nn.Linear(d_model, d_model)
        # self.key = nn.Linear(d_model, d_model)
        # self.value = nn.Linear(d_model, d_model)
        #
        # self.fc_out = nn.Linear(d_model, d_model)

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)

        # self.linear_layer = nn.Linear(max_len * d_model, 512)
        self.device = device

        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/Mol_MB_512.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        GT_state_dict = {k[len('model.GT.'):]: v for k, v in state_dict.items() if
                                  k.startswith('model.GT.')}
        # 加载到模型的 conditionEn 部分
        self.GT.load_state_dict(GT_state_dict)

        # checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/nmr_mean.ckpt')
        # # 获取模型的 state_dict
        # state_dict = checkpoint['state_dict']
        # # 从 state_dict 中提取 conditionEn 部分的权重
        # linear_layer_state_dict = {k[len('model.linear_layer.'):]: v for k, v in state_dict.items() if
        #                           k.startswith('model.linear_layer.')}
        # # 加载到模型的 conditionEn 部分
        # self.linear_layer.load_state_dict(linear_layer_state_dict)

        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/nmr_lstm_clean.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        transEn_state_dict = {k[len('model.transEn.'):]: v for k, v in state_dict.items() if
                              k.startswith('model.transEn.')}
        # 加载到模型的 conditionEn 部分
        self.transEn.load_state_dict(transEn_state_dict)

        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/nmr_lstm_clean.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        lstm_state_dict = {k[len('model.lstm.'):]: v for k, v in state_dict.items() if
                              k.startswith('model.lstm.')}
        # 加载到模型的 conditionEn 部分
        self.lstm.load_state_dict(lstm_state_dict)

        # checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/nmr_complex_attn.ckpt')
        # # 获取模型的 state_dict
        # state_dict = checkpoint['state_dict']
        # # 从 state_dict 中提取 conditionEn 部分的权重
        # transEn_state_dict = {k[len('model.transEn.'):]: v for k, v in state_dict.items() if
        #                       k.startswith('model.transEn.')}
        # # 加载到模型的 conditionEn 部分
        # self.transEn.load_state_dict(transEn_state_dict)
        #
        # # 从 state_dict 中提取 conditionEn 部分的权重
        # query_state_dict = {k[len('model.query.'):]: v for k, v in state_dict.items() if
        #                    k.startswith('model.query.')}
        # # 加载到模型的 conditionEn 部分
        # self.query.load_state_dict(query_state_dict)
        #
        # key_state_dict = {k[len('model.key.'):]: v for k, v in state_dict.items() if
        #                     k.startswith('model.key.')}
        # # 加载到模型的 conditionEn 部分
        # self.key.load_state_dict(key_state_dict)
        #
        # value_state_dict = {k[len('model.value.'):]: v for k, v in state_dict.items() if
        #                     k.startswith('model.value.')}
        # # 加载到模型的 conditionEn 部分
        # self.value.load_state_dict(value_state_dict)
        #
        # fc_out_state_dict = {k[len('model.fc_out.'):]: v for k, v in state_dict.items() if
        #                     k.startswith('model.fc_out.')}
        # # 加载到模型的 conditionEn 部分
        # self.fc_out.load_state_dict(fc_out_state_dict)


        #
        # for param in self.transEn.parameters():
        #     param.requires_grad = False
        # for param in self.GT.parameters():
        #     param.requires_grad = False
        # for param in self.linear_layer.parameters():
        #     param.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, X, E, y, node_mask, conditionVec):
        conditionVec_counts = (conditionVec != 0).sum(dim=1).cpu()
        assert isinstance(conditionVec, torch.Tensor), "conditionVec should be a tensor, but got type {}".format(
            type(conditionVec))

        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVec = self.transEn(conditionVec, srcMask)

        batch_size, seq_len, dim = conditionVec.shape
        for i in range(batch_size):
            # 获取当前样本的有效长度
            valid_length = conditionVec_counts[i]
            # 将无效部分置为 0
            conditionVec[i, valid_length:, :] = 0
        mask = (conditionVec.sum(dim=-1) != 0).float()
        output, (hidden, cell) = self.lstm(conditionVec)
        output = output * mask.unsqueeze(-1)
        last_output_idx = (mask.sum(dim=1) - 1).long()  # 计算有效时间步的索引
        batch_size = output.size(0)
        conditionVec = output[torch.arange(batch_size), last_output_idx, :]

        # batch_size, seq_len, dim = conditionVec.shape
        # for i in range(batch_size):
        #     # 获取当前样本的有效长度
        #     valid_length = conditionVec_counts[i]
        #     # 将无效部分置为 0
        #     conditionVec[i, valid_length:, :] = 0
        # packed_input = pack_padded_sequence(conditionVec, conditionVec_counts, batch_first=True,
        #                                     enforce_sorted=False)
        # # 通过LSTM处理packed sequence
        # packed_output, (hn, cn) = self.lstm(packed_input)
        # # # 如果需要可以将packed_output转换为正常的序列（这里可以跳过）
        # # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # # 使用最后的隐藏状态进行预测
        # conditionVec = hn[-1]  # 只取最后一层的输出

        # # 线性变换得到 Q, K, V
        # Q = self.query(conditionVec)  # [batch_size, seq_len, embed_dim]
        # K = self.key(conditionVec)  # [batch_size, seq_len, embed_dim]
        # V = self.value(conditionVec)  # [batch_size, seq_len, embed_dim]
        # # 将 Q, K, V 拆分为多头
        # head_num = 1
        # head_dim = 512
        # Q = Q.view(conditionVec.size(0), self.max_len, head_num, head_dim).transpose(1, 2)
        # K = K.view(conditionVec.size(0), self.max_len, head_num, head_dim).transpose(1, 2)
        # V = V.view(conditionVec.size(0), self.max_len, head_num, head_dim).transpose(1, 2)
        # # 计算注意力分数
        # attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        #     torch.tensor(head_dim, dtype=torch.float32))
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # # 加权求和
        # output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        # # 合并多头
        # output = output.transpose(1, 2).contiguous().view(conditionVec.size(0), self.max_len, self.d_model)
        # pooled_output = torch.sum(output, dim=1)
        # conditionVec = self.fc_out(pooled_output)


        # # 使用注意力权重对序列进行加权求和
        # pooled_output = torch.sum(torch.matmul(attention_weights.mean(dim=1), output), dim=1)  # [batch_size, embed_dim]
        # # 输出层
        # # pooled_output = torch.sum(output, dim=1)
        # conditionVec = self.fc_out(pooled_output)

        # conditionVec = conditionVec.view(conditionVec.size(0), -1)
        # conditionVec = self.linear_layer(conditionVec)

        # conditionVec = torch.mean(conditionVec, dim=1)

        # output, (hn, cn) = self.lstm(conditionVec)
        # conditionVec = hn.squeeze(0)


        y = torch.hstack((y, conditionVec)).float()

        return self.GT(X, E, y, node_mask)