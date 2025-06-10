import torch
from torch import nn

# from ..blocks.encoder_layer import EncoderLayer
from ..embedding.nmr_embedding import H1nmr_embedding, C13nmr_embedding

class H1nmr_encoder(nn.Module):
    def __init__(self,  d_model, dim_feedforward, n_head, num_layers, drop_prob, device):
        super().__init__()

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

        out = self.encoder(src=x_emb, mask=None, # 不适用序列掩码（通常用于解码器）
                           src_key_padding_mask=pad_mask)

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
    def __init__(self,device, dim_h=1024, dim_c=256, hidden_dim=512, out_dim=512, bi_crossattn_fusion_mode='', pool_mode='', crossmodal_fusion_mode='',):
        super().__init__()

        # 投影层
        self.proj_h = nn.Linear(dim_h, hidden_dim)
        self.proj_c = nn.Linear(dim_c, hidden_dim)
        # # 双向交叉注意力
        # self.cross_attn_ab = nn.MultiheadAttention(hidden_dim, num_heads=8)
        # self.cross_attn_ba = nn.MultiheadAttention(hidden_dim, num_heads=8)   #  batch_first=True
        #
        # self.bi_crossattn_fusion_mode = bi_crossattn_fusion_mode
        # self.pool_mode = pool_mode
        # self.crossmodal_fusion = crossmodal_fusion_mode
        #
        self.hidden_dim = hidden_dim
        # self.out_dim = out_dim
        #
        # self.gate_linear = nn.Linear(hidden_dim, 1)
        self.attn_pool = MaskedAttentionPool(dim=self.hidden_dim)

        self.device = device

    def masked_mean_pool(tensor, mask):
        # tensor: [batch, seq_len, dim]
        # mask: [batch, seq_len] (1表示有效，0表示pad)
        lengths = mask.sum(dim=1, keepdim=True)  # [batch, 1]
        masked = tensor * mask.unsqueeze(-1)  # 置零pad位置
        return masked.sum(dim=1) / (lengths + 1e-6)  # [batch, dim]

    def forward(self, tensor_Hnmr,  mask_H, tensor_Cnmr,  mask_C):

        max_len_H = mask_H.sum(dim=-1).max().item()
        mask_H = mask_H[:, :int(max_len_H)]
        max_len_C = mask_C.sum(dim=-1).max().item()
        mask_C = mask_C[:, :int(max_len_C)]

        tensor_Hnmr = tensor_Hnmr[:, :int(max_len_H), :]
        tensor_Cnmr = tensor_Cnmr[:, :int(max_len_C), :]
        # 投影到统一维度
        H_aligned = self.proj_h(tensor_Hnmr)
        C_aligned = self.proj_c(tensor_Cnmr)

        fused_H = H_aligned
        fused_C = C_aligned

        '''---------------------------------------------------------------------------------'''

        '''模态内聚合----------------------------'''

        # 对两个模态分别做注意力池化
        # pool = MaskedAttentionPool(dim=self.hidden_dim)
        global_H = self.attn_pool(fused_H, mask_H)
        global_C = self.attn_pool(fused_C, mask_C)  # [batch, 256]
        #
        return global_H, global_C




class NMR_encoder(nn.Module):

    def __init__(self,  device, dim_H, dimff_H, dim_C, dimff_C, hidden_dim, n_head, num_layers, drop_prob):
        super().__init__()
        self.H1nmr_encoder = H1nmr_encoder(d_model=dim_H, dim_feedforward=dimff_H, n_head=n_head, num_layers=num_layers, drop_prob=drop_prob, device=device)

        self.C13nmr_encoder = C13nmr_encoder(d_model=dim_C, dim_feedforward=dimff_C, n_head=n_head, num_layers=num_layers, drop_prob=drop_prob, device=device)

        self.NMR_fusion = NMR_fusion(device, dim_H,dim_C,hidden_dim, bi_crossattn_fusion_mode='gated', pool_mode='attn_pool', crossmodal_fusion_mode='weighted_sum')

        self.device = device

    def create_mask(self, batch_size, max_seq_len, num_peak):

        mask = torch.zeros(batch_size, max_seq_len, device=self.device)
        for i, length in enumerate(num_peak):
            mask[i, :length] = 1
        return mask

    def forward(self, condition):
        H1nmr, num_H_peak, C13nmr, num_C_peak = condition

        batch_size, max_seq_len_H, _ = H1nmr.size()
        mask_H = self.create_mask(batch_size, max_seq_len_H, num_H_peak)
        _, max_seq_len_C = C13nmr.size()
        mask_C = self.create_mask(batch_size, max_seq_len_C, num_C_peak)

        h_feat = self.H1nmr_encoder(H1nmr, mask_H)  # [batch, h_seq, h_dim]
        c_feat = self.C13nmr_encoder(C13nmr, mask_C)  # [batch, c_seq, c_dim]

        global_H, global_C = self.NMR_fusion(h_feat, mask_H, c_feat, mask_C)  # [batch, fusion_dim]

        return global_H, global_C