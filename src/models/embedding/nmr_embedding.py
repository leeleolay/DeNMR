import torch
from torch import nn



class RBFEncoder(nn.Module):
    def __init__(self, min, max, bins):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(min, max, bins), requires_grad=False)
        self.sigma = (max - min) / (bins-1)  # 自适应带宽
        # print('')

    def forward(self, x):
        # x: (...,)
        diff = x.unsqueeze(-1) - self.centers  # (..., bins)
        return torch.exp(-0.5 * (diff / self.sigma).pow(2))


class RBFEncoder_Jcouple(nn.Module):
    def __init__(self, min1=0, max1=26, bins1=131, min2=27, max2=58, bins2=32):
        super().__init__()


        centers1 = torch.linspace(min1, max1, bins1)
        sigma1 = (max1 - min1) / (bins1 - 1)  # 20/99 ≈ 0.202

        centers2 = torch.linspace(min2, max2, bins2)
        sigma2 = (max2 - min2) / (bins2 - 1)  # 30/29 ≈ 1.034

        # 合并参数
        self.centers = nn.Parameter(
            torch.cat([centers1, centers2]),
            requires_grad=False
        )
        self.sigma = nn.Parameter(
            torch.cat([
                torch.full((bins1,), sigma1),
                torch.full((bins2,), sigma2)
            ]),
            requires_grad=False
        )

        # print('')

    def forward(self, x):
        diff = x.unsqueeze(-1) - self.centers  # (..., 130)
        return torch.exp(-0.5 * (diff / self.sigma).pow(2))


# class RBFEncoder_learnable(nn.Module):
#     def __init__(self, num_centers=61, init_centers=None, gamma=0.2):
#         super().__init__()
#         if init_centers is None:
#             init_centers = torch.linspace(-1, 10, num_centers)  # 假设特征范围0-3
#         self.centers = nn.Parameter(init_centers.view(1, 1, -1))  # [1,1,num_centers]
#         self.gamma = nn.Parameter(torch.tensor(gamma))
#
#     def forward(self, x):
#         """x: [batch, seq_len, 1]"""
#         diff = x - self.centers  # 广播计算 [batch, seq_len, num_centers]
#         return torch.exp(-self.gamma * (diff ** 2))

class H1nmr_embedding(nn.Module):
    def __init__(self,  device, split_dim=64, peakwidth_dim=40, integral_dim=32,
                 H_shift_min=-1, H_shift_max=10, H_shift_bin=111,
                 min_j=0, max_j=58, j_bins1=131, j_bins2=32, hidden=1024, dim=1024, drop_prob=0.1):
        super().__init__()

        self.shift_emb = RBFEncoder(min=H_shift_min, max=H_shift_max, bins=H_shift_bin)  # 覆盖常见1H范围

        self.peakwidth_emb = nn.Embedding(70, peakwidth_dim, padding_idx=0) #70

        self.split_emb = nn.Embedding(116, split_dim, padding_idx=0)  # 支持116种裂分模式

        self.integral_emb = nn.Embedding(26, integral_dim, padding_idx=0) #26

        self.J_emb = RBFEncoder_Jcouple(min1=min_j, max1=26, bins1=j_bins1, min2=27, max2=max_j, bins2=j_bins2)

        self.d_model = split_dim+peakwidth_dim+integral_dim+H_shift_bin+j_bins1+j_bins2

        self.peak_fuser = peak_fuser(self.d_model, dim, drop_prob)

        self.device = device


    def forward(self, h1nmr, src_mask):

        hnmr = h1nmr.to(self.device)

        h_shift, peakwidth, split, integral, j_couple = hnmr[:,:,0], hnmr[:,:,1], hnmr[:,:,2], hnmr[:,:,3], hnmr[:,:,4:]

        h_shift_emb = self.shift_emb(h_shift) * src_mask.unsqueeze(-1)
        peakwidth_emb = self.peakwidth_emb(peakwidth.long())
        split_emb = self.split_emb(split.long())
        integral_emb = self.integral_emb((integral+1).long())

        J_emb = self.J_emb(j_couple)
        J_emb = torch.sum(J_emb, dim=-2) * src_mask.unsqueeze(-1)

        hnmr_emb = torch.cat([h_shift_emb, peakwidth_emb, split_emb, integral_emb, J_emb], dim=-1)
        hnmr_emb = self.peak_fuser(hnmr_emb)

        return hnmr_emb


class C13nmr_embedding(nn.Module):
    def __init__(self, device, C_shift_min=-15, C_shift_max=229, C_bins=245,  hidden=512, dim=256, drop_prob=0.1):
        super().__init__()

        self.shift_emb = RBFEncoder(min=C_shift_min, max=C_shift_max, bins=C_bins)

        # self.PositionwiseFeedForward = PositionwiseFeedForward(C_bins, hidden, dim, drop_prob)
        self.peak_fuser = peak_fuser(C_bins, dim, drop_prob)


        self.device = device

    def forward(self, c13nmr, src_mask):

        cnmr = c13nmr.to(self.device)

        c_shift_emb = self.shift_emb(cnmr) * src_mask.unsqueeze(-1)

        # cnmr_emb = self.PositionwiseFeedForward(c_shift_emb)
        cnmr_emb = self.peak_fuser(c_shift_emb)

        return cnmr_emb


class peak_fuser(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()  # 更简洁的super调用
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop_prob))

        # PositionwiseFeedForward
        # self.net = nn.Sequential(
        #     nn.Linear(d_model, hidden),
        #     nn.GELU(),
        #     nn.Dropout(drop_prob),
        #     nn.Linear(hidden, d_model))

    def forward(self, x):
        return self.net(x)

