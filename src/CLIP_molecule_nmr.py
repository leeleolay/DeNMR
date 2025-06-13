import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.contrastGT import contrastGT
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from src import utils


class CLIP_molecule_nmr(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        # self.enc_voc_size = 5450
        # self.max_len = 256
        # self.d_model = 256
        # self.ffn_hidden = 1024
        # self.n_head = 8
        # self.n_layers_TE = 3
        # self.drop_prob = 0.

        self.enc_voc_size = 5450
        # self.max_len = 256
        # self.d_model = 256
        self.dim_enc_H = 1024
        self.dimff_enc_H = 2048
        self.dim_enc_C = 256
        self.dimff_enc_C = 512

        self.ffn_hidden = 512
        self.n_head = 8
        self.n_layers_TE = 3
        self.drop_prob = 0.
        device = torch.device("cuda")

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.tem = 2
        self.val_loss = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features


        self.model = contrastGT(n_layers_GT=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU(),
                                      dim_enc_H=self.dim_enc_H,
                                      dimff_enc_H=self.dimff_enc_H,
                                      dim_enc_C=self.dim_enc_C,
                                      dimff_enc_C=self.dimff_enc_C,
                                      ffn_hidden=self.ffn_hidden, n_head=self.n_head, n_layers_TE=self.n_layers_TE, drop_prob=self.drop_prob, device=device)

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256
        self.seq_len_H1 = 20
        self.seq_len_C13 = 75

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        condition_H1nmr = data.H_nmr
        condition_H1nmr = condition_H1nmr.reshape(batch_length, self.seq_len_H1, -1)
        condition_C13nmr = data.C_nmr
        condition_C13nmr = condition_C13nmr.reshape(batch_length, self.seq_len_C13)
        num_H_peak = data.num_H_peak
        num_C_peak = data.num_C_peak
        conditionAll = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]

        predV1, predV2 = self.forward(noisy_data, extra_data, node_mask, X, E, conditionAll)

        V1_f = predV1  # 假设 V1 是从图像（或者其他模态）得到的特征
        V2_f = predV2  # 假设 V2 是从文本（或者其他模态）得到的特征

        # L2 归一化
        V1_e = F.normalize(V1_f, p=2, dim=1)  # 对 V1_f 进行 L2 归一化
        V2_e = F.normalize(V2_f, p=2, dim=1)  # 对 V2_f 进行 L2 归一化

        # 计算缩放的余弦相似度
        logits = torch.matmul(V1_e, V2_e.T) * torch.exp(torch.tensor(self.tem, device=V1_e.device))

        # 生成标签
        n = V1_f.size(0)  # 假设 batch_size = 512
        labels = torch.arange(n, device=V1_f.device)  # 对角线的标签

        # 计算对比损失
        loss_fn = torch.nn.CrossEntropyLoss()

        # 计算对比损失：注意 logits 的维度，交叉熵损失会自动考虑标签
        loss_v1 = loss_fn(logits, labels)
        loss_v2 = loss_fn(logits.T, labels)  # 对称计算

        # 对称的对比学习损失
        loss = (loss_v1 + loss_v2) / 2

        # # 计算 Vec = predV1 - predV2
        # Vec = (predV1 - predV2) ** 2
        # Vec_squared_sum = Vec.sum(dim=1)  # 或者 Vec ** 2 后再 sum(dim=1)
        # # 对所有样本的平方和求平均
        # loss = Vec_squared_sum.mean()

        #if i%100 == 0:
            # print(f'train_predV1{predV1}')
            # print(f'train_predV2{predV2}')
            # print(f'train_Vec_squared_sum{Vec_squared_sum}')
        print(f"train_loss:{loss}")
        sys.stdout.flush()
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        torch.save(self.state_dict(), "/home/liuxuwei01/PaddleMaterial/output/step2_init_weights.pth")

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")

    def on_train_epoch_end(self) -> None:
        sys.stdout.flush()

    def on_validation_epoch_start(self) -> None:
        self.val_loss = []


    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        condition_H1nmr = data.H_nmr
        condition_H1nmr = condition_H1nmr.reshape(batch_length, self.seq_len_H1, -1)
        condition_C13nmr = data.C_nmr
        condition_C13nmr = condition_C13nmr.reshape(batch_length, self.seq_len_C13)
        num_H_peak = data.num_H_peak
        num_C_peak = data.num_C_peak
        conditionAll = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]

        predV1, predV2 = self.forward(noisy_data, extra_data, node_mask, X, E, conditionAll)

        V1_f = predV1  # 假设 V1 是从图像（或者其他模态）得到的特征
        V2_f = predV2  # 假设 V2 是从文本（或者其他模态）得到的特征

        # L2 归一化
        V1_e = F.normalize(V1_f, p=2, dim=1)  # 对 V1_f 进行 L2 归一化
        V2_e = F.normalize(V2_f, p=2, dim=1)  # 对 V2_f 进行 L2 归一化

        # 计算缩放的余弦相似度
        logits = torch.matmul(V1_e, V2_e.T) * torch.exp(torch.tensor(self.tem, device=V1_e.device))

        # 生成标签
        n = V1_f.size(0)  # 假设 batch_size = 512
        labels = torch.arange(n, device=V1_f.device)  # 对角线的标签

        # 计算对比损失
        loss_fn = torch.nn.CrossEntropyLoss()

        # 计算对比损失：注意 logits 的维度，交叉熵损失会自动考虑标签
        loss_v1 = loss_fn(logits, labels)
        loss_v2 = loss_fn(logits.T, labels)  # 对称计算

        # 对称的对比学习损失
        loss = (loss_v1 + loss_v2) / 2

        # # 计算 Vec = predV1 - predV2
        # Vec = (predV1 - predV2) ** 2
        #
        # Vec_squared_sum = Vec.sum(dim=1)
        #
        # # 对所有样本的平方和求平均
        # loss = Vec_squared_sum.mean()
        self.val_loss.append(loss)

        if i%8 == 0:
            # print(f'val_predV1{predV1}')
            # print(f'val_predV2{predV2}')
            # print(f'val_Vec_squared_sum{Vec_squared_sum}')
            print(f"val_loss:{loss}")

        print(f"val_loss:{loss}")
        # nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, condition=conditionAll, test=False)
        return {'loss': loss}

    def on_validation_epoch_end(self) -> None:
        val_loss = sum(self.val_loss)
        self.log("val_loss", val_loss, sync_dist=True)
        self.print(f"Epoch {self.current_epoch}: Val Loss {val_loss :.2f} ")


    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, data, i):
        pass

    def on_test_epoch_end(self) -> None:
        self.print("Done testing.")

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data


    def forward(self, noisy_data, extra_data, node_mask, X, E, condition):
        xtrue = X
        etrue = E
        X_ = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E_ = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y_ = torch.hstack((noisy_data['y_t'], extra_data.y)).float()

        # condition = torch.tensor(condition)
        # condition = condition.clone().detach().requires_grad_(True)
        # 生成 [batch] 大小的随机概率向量，每个元素以 15% 的概率小于 0.15
        # 将需要置为 0 的行设置为 0
        return self.model(X_, E_, y_, node_mask, xtrue, etrue, condition)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
