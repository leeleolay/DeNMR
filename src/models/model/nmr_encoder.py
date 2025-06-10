import torch
from torch import nn

# from ..blocks.encoder_layer import EncoderLayer
from ..embedding.nmr_embedding import H1nmr_embedding, C13nmr_embedding

# class H1nmr_encoder(nn.Module):
#     def __init__(self,  d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
#         super().__init__()
#
#         self.embed = H1nmr_embedding(device)
#         # Transformer
#         self.trans_encoder = nn.ModuleList([EncoderLayer(d_model=d_model,
#                                                          ffn_hidden=ffn_hidden,
#                                                          n_head=n_head,
#                                                          drop_prob=drop_prob)
#                                             for _ in range(n_layers)])
#
#     def forward(self, x, src_mask):
#         # 输入格式：[batch, len_peak, feat_dim]
#         x = self.embed(x, src_mask)
#
#         for trans_layer in self.trans_encoder:
#             x = trans_layer(x, src_mask)
#
#         return x

# class C13nmr_encoder(nn.Module):
#     def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
#         super().__init__()
#
#         self.embed = C13nmr_embedding(device)
#         # Transformer
#         self.trans_encoder = nn.ModuleList([EncoderLayer(d_model=d_model,
#                                                          ffn_hidden=ffn_hidden,
#                                                          n_head=n_head,
#                                                          drop_prob=drop_prob)
#                                             for _ in range(n_layers)])
#
#     def forward(self, x, src_mask):
#         # 输入格式：[batch, len_peak, feat_dim]
#         x = self.embed(x, src_mask)
#
#         for trans_layer in self.trans_encoder:
#             x = trans_layer(x, src_mask)
#
#         return x

class H1nmr_encoder(nn.Module):
    def __init__(self,  d_model, dim_feedforward, n_head, num_layers, drop_prob, device):
        super().__init__()

        self.num_heads = n_head
        self.embed = H1nmr_embedding(device, dim=d_model,drop_prob=drop_prob)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True  # 使用batch_first格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask):
        # 输入格式：[batch, len_peak, feat_dim]
        x_emb = self.embed(x, src_mask)

        pad_mask = (src_mask == 0)

        bsz, src_len, _ = x_emb.shape
        pad_mask_new = pad_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, src_len, -1).reshape(bsz * self.num_heads, src_len, src_len)
        out = self.encoder(src=x_emb, mask=pad_mask_new)
        
        # out = self.encoder(src=x_emb, mask=None, # 不适用序列掩码（通常用于解码器）
        #                    src_key_padding_mask=pad_mask)

        return out

class C13nmr_encoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, n_head, num_layers, drop_prob, device):
        super().__init__()

        self.embed = C13nmr_embedding(device,dim=d_model,drop_prob=drop_prob)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True  # 使用batch_first格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask):
        # 输入格式：[batch, len_peak, feat_dim]
        x_emb = self.embed(x, src_mask)

        pad_mask = (src_mask == 0)

        out = self.encoder(src=x_emb, mask=None,  # 不适用序列掩码（通常用于解码器）
                           src_key_padding_mask=pad_mask)

        return out

# # 方法3：加权平均（可学习注意力）
# class AttentionPool(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x, mask):
#         # x: [batch, seq_len, dim]
#         attn_weights = self.attention(x)  # [batch, seq_len, 1]
#         attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -float('inf'))
#         attn_weights = torch.softmax(attn_weights, dim=1)
#         return (x * attn_weights).sum(dim=1)  # [batch, dim]


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )  # 移除了Softmax，需手动处理

    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        # mask: [batch, seq_len] （1表示有效，0表示pad）
        attn_scores = self.attention(x)  # [batch, seq_len, 1]

        # === 新增：掩码处理 ===
        if mask is not None:
            # 将padding位置的注意力分数设为-∞，Softmax后权重为0
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0,-float('inf'))


        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        return (x * attn_weights).sum(dim=1)  # [batch, dim]


class MaskedCrossModalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim))

    def forward(self, x, mask):
        # x: [batch, seq, dim], mask: [batch, seq] (1有效，0pad)
        attn_scores = torch.matmul(x, self.query.T)  # [batch, seq, 1]

        # 掩码处理：将pad位置的分数设为-∞
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -float('inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)

        return (x * attn_weights).sum(dim=1)  # [batch, dim]


class NMR_fusion(nn.Module):
    def __init__(self,device, dim_h=1024, dim_c=256, hidden_dim=512, num_heads=8, out_dim=512, bi_crossattn_fusion_mode='', pool_mode='', crossmodal_fusion_mode='',):
        super().__init__()

        # 投影层
        self.proj_h = nn.Linear(dim_h, hidden_dim)
        self.proj_c = nn.Linear(dim_c, hidden_dim)
        # 双向交叉注意力
        self.cross_attn_ab = nn.MultiheadAttention(hidden_dim, num_heads)
        self.cross_attn_ba = nn.MultiheadAttention(hidden_dim, num_heads)

        self.bi_crossattn_fusion_mode = bi_crossattn_fusion_mode
        self.pool_mode = pool_mode
        self.crossmodal_fusion = crossmodal_fusion_mode

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.gate_linear = nn.Linear(hidden_dim, 1)
        # self.pool = MaskedCrossModalAttentionPool(dim=self.hidden_dim)
        self.attn_pool = MaskedAttentionPool(dim=self.hidden_dim)
        self.weighted_sum = nn.Linear(1024, 1)
        self.concat_linear = nn.Linear(1024,512)

        self.device = device

    def masked_mean_pool(self, tensor, mask):
        # tensor: [batch, seq_len, dim]
        # mask: [batch, seq_len] (1表示有效，0表示pad)
        lengths = mask.sum(dim=1, keepdim=True)  # [batch, 1]
        masked = tensor * mask.unsqueeze(-1)  # 置零pad位置
        return masked.sum(dim=1) / (lengths + 1e-6)  # [batch, dim]

    def forward(self, tensor_Hnmr,  mask_H, tensor_Cnmr,  mask_C):

        max_len_H = mask_H.sum(dim=-1).max().item()
        mask_H = mask_H[:, :int(max_len_H)]
        tensor_Hnmr = tensor_Hnmr[:, :int(max_len_H), :]
        max_len_C = mask_C.sum(dim=-1).max().item()
        mask_C = mask_C[:, :int(max_len_C)]
        tensor_Cnmr = tensor_Cnmr[:, :int(max_len_C), :]

        # 投影到统一维度
        H_aligned = self.proj_h(tensor_Hnmr)
        C_aligned = self.proj_c(tensor_Cnmr)

        H_aligned_perm = H_aligned.permute(1, 0, 2)  # [seq_a, batch, hidden_dim]
        C_aligned_perm = C_aligned.permute(1, 0, 2)  # [seq_b, batch, hidden_dim]

        # print(f'mask_H{mask_H.shape}')

        # max_len_H = mask_H.sum(dim=-1).max().item()
        # mask_H = mask_H[:, :int(max_len_H)]
        # H_aligned_perm = H_aligned_perm[:int(max_len_H), :, :]
        # max_len_C = mask_C.sum(dim=-1).max().item()
        # mask_C = mask_C[:, :int(max_len_C)]
        # C_aligned_perm = C_aligned_perm[:int(max_len_C), :, :]

        # 双向交叉注意力
        pad_mask_H = (mask_H == 0)
        pad_mask_C = (mask_C == 0)

        # print(f'H_aligned_perm{H_aligned_perm.shape}')
        # print(f'C_aligned_perm{C_aligned_perm.shape}')
        # print(f'pad_mask_C{pad_mask_C.shape}')

        attn_H2C, _ = self.cross_attn_ab(query=H_aligned_perm, key=C_aligned_perm, value=C_aligned_perm, key_padding_mask=pad_mask_C)
        attn_C2H, _ = self.cross_attn_ba(query=C_aligned_perm, key=H_aligned_perm, value=H_aligned_perm, key_padding_mask=pad_mask_H)

        attn_H2C = attn_H2C.permute(1, 0, 2)  # [batch, seq_a, hidden_dim]
        attn_C2H = attn_C2H.permute(1, 0, 2)

        '''两个模态的交叉注意力输出与原始特征结合------------------------'''

        if self.bi_crossattn_fusion_mode == 'concat':
            # 方式1：拼接两个方向的输出
            fused_H = torch.cat([H_aligned, attn_H2C], dim=-1)  # [batch, seq_a, 2*hidden_dim]
            fused_C = torch.cat([C_aligned, attn_C2H], dim=-1)  # [batch, seq_b, 2*hidden_dim]

        elif self.bi_crossattn_fusion_mode == 'add':
            # 方式2：残差连接（保留原始信息）
            fused_H = H_aligned + attn_H2C  # [batch, seq_a, hidden_dim]
            fused_C = C_aligned + attn_C2H  # [batch, seq_b, hidden_dim]

        elif self.bi_crossattn_fusion_mode == 'gated':
            # 方式3：门控融合（自适应权重）
            # gate_H = torch.sigmoid(nn.Linear(self.hidden_dim, 1)(attn_H2C))  # 计算权重
            gate_H = torch.sigmoid(self.gate_linear(attn_H2C))
            fused_H = (1 - gate_H) * H_aligned + gate_H * attn_H2C
            gate_C = torch.sigmoid(self.gate_linear(attn_C2H)) # 计算权重
            fused_C = (1 - gate_C) * C_aligned + gate_C * attn_C2H

        else:
            fused_H = attn_H2C
            fused_C = attn_C2H

        '''---------------------------------------------------------------------------------'''

        '''模态内聚合----------------------------'''

        if self.pool_mode == 'mean_pool':
            # 方法1：平均池化
            global_H = self.masked_mean_pool(fused_H, mask_H)

            global_C = self.masked_mean_pool(fused_C, mask_C)  # [batch, 256]

        elif self.pool_mode == 'attn_pool':
            # 对两个模态分别做注意力池化
            # pool = MaskedAttentionPool(dim=self.hidden_dim)
            global_H = self.attn_pool(fused_H, mask_H)
            global_C = self.attn_pool(fused_C, mask_C)  # [batch, 256]

        '''跨模态融合---------------------------------'''

        if self.crossmodal_fusion == 'concat_linear':
            merged = torch.cat([global_H, global_C], dim=-1)  # [batch, 512]
            global_output = self.concat_linear(merged)  # 压缩到 [batch, 256]

        elif self.crossmodal_fusion == 'weighted_sum':
            merged = torch.cat([global_H, global_C], dim=-1)
            # 或者
            # merged = global_H + global_C

            gate = torch.sigmoid(self.weighted_sum(merged))  # [batch,1]
            global_output = gate * global_H + (1 - gate) * global_C

        '''--------------------------------------------------------------------------------------------------'''

        '''直接跨模态池化'''

        # combined_features = torch.cat([fused_H, fused_C], dim=1)
        # # pool = MaskedCrossModalAttentionPool(dim=self.hidden_dim)
        # mask = torch.cat([mask_H, mask_C], dim=-1)
        # global_output = self.pool(combined_features,mask)

        return global_output




class NMR_encoder(nn.Module):

    def __init__(self,  device, dim_H, dimff_H, dim_C, dimff_C, hidden_dim, n_head, num_layers, drop_prob):
        super().__init__()
        self.H1nmr_encoder = H1nmr_encoder(d_model=dim_H, dim_feedforward=dimff_H, n_head=n_head, num_layers=num_layers, drop_prob=drop_prob, device=device)

        self.C13nmr_encoder = C13nmr_encoder(d_model=dim_C, dim_feedforward=dimff_C, n_head=n_head, num_layers=num_layers, drop_prob=drop_prob, device=device)

        self.NMR_fusion = NMR_fusion(device, dim_H,dim_C,hidden_dim, bi_crossattn_fusion_mode='add', pool_mode='attn_pool', crossmodal_fusion_mode='concat_linear')

        self.device = device

    def create_mask(self, batch_size, max_seq_len, num_peak):

        mask = torch.zeros(batch_size, max_seq_len, device=self.device)
        for i, length in enumerate(num_peak):
            mask[i, :int(length)] = 1
        return mask

    def forward(self, condition):
        H1nmr, num_H_peak, C13nmr, num_C_peak = condition

        batch_size, max_seq_len_H, _ = H1nmr.size()
        mask_H = self.create_mask(batch_size, max_seq_len_H, num_H_peak)
        _, max_seq_len_C = C13nmr.size()
        mask_C = self.create_mask(batch_size, max_seq_len_C, num_C_peak)

        h_feat = self.H1nmr_encoder(H1nmr, mask_H)  # [batch, h_seq, h_dim]
        c_feat = self.C13nmr_encoder(C13nmr, mask_C)  # [batch, c_seq, c_dim]

        fused_feat = self.NMR_fusion(h_feat, mask_H, c_feat, mask_C)  # [batch, fusion_dim]

        return fused_feat