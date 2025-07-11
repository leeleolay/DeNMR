import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


# from models.conditionGT import ConditionGT
from models.conditionGT_new import ConditionGT
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils

import rdkit.Chem
import wandb

from rdkit.Chem import DataStructs
from rdkit.Chem import RDKFingerprint

class DiscreteDenoisingDiffusionCondition(pl.LightningModule):
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
        # self.guide_scale = 4

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        # self.val_y_collection = []
        self.val_y_condition_H1nmr = []
        self.val_y_condition_C13nmr = []
        self.val_y_num_H_peak = []
        self.val_y_num_C_peak = []
        self.val_atomCount = []
        self.val_x = []
        self.val_e = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        # self.test_y_collection = []
        self.test_y_condition_H1nmr = []
        self.test_y_condition_C13nmr = []
        self.test_y_num_H_peak = []
        self.test_y_num_C_peak = []
        self.test_atomCount = []
        self.test_x = []
        self.test_e = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features
        
        self.epoch_idx = 0
        
        self._epoch_losses = []
        self._eval_epoch_losses = []

        # self.model = ConditionGT(n_layers_GT=cfg.model.n_layers,
        #                               input_dims=input_dims,
        #                               hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        #                               hidden_dims=cfg.model.hidden_dims,
        #                               output_dims=output_dims,
        #                               act_fn_in=nn.ReLU(),
        #                               act_fn_out=nn.ReLU(),enc_voc_size=self.enc_voc_size, max_len=self.max_len, d_model=self.d_model,
        #                               ffn_hidden=self.ffn_hidden, n_head=self.n_head, n_layers_TE=self.n_layers_TE,
        #                               drop_prob=self.drop_prob, device=device)

        self.model = ConditionGT(n_layers_GT=cfg.model.n_layers,
                                 input_dims=input_dims,
                                 hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                 hidden_dims=cfg.model.hidden_dims,
                                 output_dims=output_dims,
                                 act_fn_in=nn.ReLU(),
                                 act_fn_out=nn.ReLU(),
                                 dim_enc_H=self.dim_enc_H, dimff_enc_H=self.dimff_enc_H, dim_enc_C=self.dim_enc_C, dimff_enc_C=self.dimff_enc_C,
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

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        '''********'''
        # self.vocabDim = 256
        self.seq_len_H1 = 20
        self.seq_len_C13 = 75
        '''_________'''

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
        '''*******'''
        # conditionAll = data.conditionVec
        # conditionAll = conditionAll.reshape(batch_length, self.vocabDim)
        condition_H1nmr = data.H_nmr
        condition_H1nmr = condition_H1nmr.reshape(batch_length, self.seq_len_H1, -1)

        condition_C13nmr = data.C_nmr
        condition_C13nmr = condition_C13nmr.reshape(batch_length, self.seq_len_C13)

        num_H_peak = data.num_H_peak
        num_C_peak = data.num_C_peak

        conditionVec = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]
        '''________________'''
        ######## TODO add new
        input_X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        input_E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        input_y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        ####################
        pred = self.forward(input_X, input_E, input_y, node_mask, conditionVec)
        # pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)
        self._epoch_losses.append(loss.detach())
        #if i%80 == 0:
        print(f"step:{i+1}(Train): train_loss:{loss}")
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)
        sys.stdout.flush()
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print(f"Starting train epoch... Epoch {self.epoch_idx}")
        self.epoch_idx = self.epoch_idx + 1
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        self._epoch_losses.clear()

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self._epoch_losses).mean()
        to_log = self.train_loss.log_epoch_metrics()
        elapsed = time.time() - self.start_epoch_time
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        at_str   = " | ".join(f"{k}: {v:.3f}" for k, v in epoch_at_metrics.items())
        bond_str = " | ".join(f"{k}: {v:.3f}" for k, v in epoch_bond_metrics.items())
        self.print(
            f"Epoch {self.current_epoch}(Train): "
            f"Loss: {avg_loss:.3f} | "
            f"X_CE: {to_log['train_epoch/x_CE']:.3f} | "
            f"E_CE: {to_log['train_epoch/E_CE']:.3f} | "
            f"y_CE: {to_log['train_epoch/y_CE']:.3f} | "
            f"{at_str} | {bond_str} | "
            f"{elapsed:.1f}s"
        )
        # self.print(f"Epoch {self.current_epoch}(Train): X_CE: {to_log['train_epoch/x_CE'] :.3f} | E_CE: {to_log['train_epoch/E_CE'] :.3f} | y_CE: {to_log['train_epoch/y_CE'] :.3f} | {time.time() - self.start_epoch_time:.1f}s ")
        # self.print(f"Epoch {self.current_epoch}(Train): {epoch_at_metrics} | {epoch_bond_metrics}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
        # self.val_y_collection = []
        self.val_y_condition_H1nmr = []
        self.val_y_condition_C13nmr = []
        self.val_y_num_H_peak = []
        self.val_y_num_C_peak = []
        self.val_atomCount = []
        self.val_x = []
        self.val_e= []
        self._eval_epoch_losses.clear()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        '''********'''
        # conditionAll = data.conditionVec
        # conditionAll = conditionAll.reshape(batch_length, self.vocabDim)
        condition_H1nmr = data.H_nmr
        condition_H1nmr = condition_H1nmr.reshape(batch_length, self.seq_len_H1, -1)

        condition_C13nmr = data.C_nmr
        condition_C13nmr = condition_C13nmr.reshape(batch_length, self.seq_len_C13)

        num_H_peak = data.num_H_peak
        num_C_peak = data.num_C_peak

        conditionVec = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]
        '''__________________'''

        input_X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        input_E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        input_y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        ####################
        pred = self.forward(input_X, input_E, input_y, node_mask, conditionVec)
        # self.val_y_collection.append(data.conditionVec)
        self.val_y_condition_H1nmr.append(condition_H1nmr)
        self.val_y_condition_C13nmr.append(condition_C13nmr)
        self.val_y_num_H_peak.append(num_H_peak)
        self.val_y_num_C_peak.append(num_C_peak)
        self.val_atomCount.append(data.atom_count)
        self.val_x.append(X)
        self.val_e.append(E)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)
        # if i % 10 == 0:
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, condition=conditionVec, test=False)
        print((f"step {i}(Eval): val_loss:{loss}  val_nll:{nll}"))
        self._eval_epoch_losses.append(loss.detach())
        return {"loss": loss, 'nll': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        avg_eval_loss = torch.stack(self._eval_epoch_losses).mean()
        self.print(f"Epoch {self.current_epoch}(Eval): Val Loss {avg_eval_loss :.2f} | Val NLL {metrics[0] :.2f} | Val Atom type KL {metrics[1] :.2f} | Val Edge type KL: {metrics[2] :.2f}")
        sys.stdout.flush()
        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        # self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples, all_ys, ident = [], [], 0

            # self.val_y_collection = torch.cat(self.val_y_collection, dim=0)
            self.val_atomCount = torch.cat(self.val_atomCount, dim=0)
            # self.val_y_collection = self.val_y_collection.view(-1,self.vocabDim)
            self.val_y_condition_H1nmr = torch.cat(self.val_y_condition_H1nmr, dim=0)
            self.val_y_condition_C13nmr = torch.cat(self.val_y_condition_C13nmr, dim=0)
            self.val_y_num_H_peak = torch.cat(self.val_y_num_H_peak, dim=0)
            self.val_y_num_C_peak = torch.cat(self.val_y_num_C_peak, dim=0)
            self.val_x = torch.cat(self.val_x, dim=0)
            self.val_e = torch.cat(self.val_e, dim=0)
            # num_examples = self.val_y_collection.size(0)
            num_examples = self.val_y_condition_H1nmr.size(0)

            start_index = 0

            self.val_allnum = len(self.val_x)
            self.val_right = 0

            while samples_left_to_generate > 0:
                bs = 1 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                if start_index + to_generate > num_examples:
                    start_index = 0
                if to_generate > num_examples:
                    ratio = to_generate // num_examples
                    # self.val_y_collection = self.val_y_collection.repeat(ratio+1, 1)
                    self.val_y_condition_H1nmr = self.val_y_condition_H1nmr.repeat(ratio+1, 1)
                    self.val_y_condition_C13nmr = self.val_y_condition_C13nmr.repeat(ratio+1, 1)
                    self.val_y_num_H_peak = self.val_y_num_H_peak.repeat(ratio+1, 1)
                    self.val_y_num_C_peak = self.val_y_num_C_peak.repeat(ratio+1, 1)
                    self.val_atomCount = self.val_atomCount.repeat(ratio + 1, 1)
                    self.val_x = self.val_x.repeat(ratio + 1, 1)
                    self.val_e = self.val_e.repeat(ratio + 1, 1)
                    # num_examples = self.val_y_collection_H1nmr.size(0)
                    num_examples = self.val_y_condition_H1nmr.size(0)
                # batch_y = self.val_y_collection[start_index:start_index + to_generate]
                batch_y_condition_H1nmr = self.val_y_condition_H1nmr[start_index:start_index + to_generate]
                batch_y_condition_C13nmr = self.val_y_condition_C13nmr[start_index:start_index + to_generate]
                batch_y_num_H_peak = self.val_y_num_H_peak[start_index:start_index + to_generate]
                batch_y_num_C_peak = self.val_y_num_C_peak[start_index:start_index + to_generate]
                batch_y = [batch_y_condition_H1nmr, batch_y_num_H_peak, batch_y_condition_C13nmr, batch_y_num_C_peak]
                batch_atomCount = self.val_atomCount[start_index:start_index + to_generate]
                batch_x = self.val_x[start_index:start_index + to_generate]
                batch_e = self.val_e[start_index:start_index + to_generate]

                molecule_list, molecule_list_True = self.sample_batch(batch_id=ident,
                                                                      batch_size=to_generate,
                                                                      num_nodes=batch_atomCount,
                                                                      batch_condition=batch_y,
                                                                      save_final=to_save,
                                                                      keep_chain=chains_save,
                                                                      number_chain_steps=self.number_chain_steps,
                                                                      batch_X=batch_x,
                                                                      batch_E=batch_e)

                samples.extend(molecule_list)

                # samples.extend(self.sample_batch(batch_id=ident,
                #                                  batch_size=to_generate,
                #                                  num_nodes=batch_atomCount,
                #                                  batch_condition = batch_y,
                #                                  batch_x =batch_x,
                #                                  batch_e =batch_e,
                #                                  save_final=to_save,
                #                                  keep_chain=chains_save,
                #                                  number_chain_steps=self.number_chain_steps))

                for i in range(to_generate):
                    mol = self.mol_from_graphs(molecule_list[i][0].numpy(), molecule_list[i][1].numpy())
                    mol_true = self.mol_from_graphs(molecule_list_True[i][0].numpy(), molecule_list_True[i][1].numpy())
                    try:
                        fp1 = RDKFingerprint(mol)
                        fp2 = RDKFingerprint(mol_true)
                        # 计算Tanimoto相似度
                        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                        # 输出相似度
                        if similarity == 1:
                            self.val_right = self.val_right + 1
                    except rdkit.Chem.KekulizeException:
                        print("Can't kekulize molecule")

                ident += to_generate
                start_index += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.print(f'right:{self.val_right}')
            self.print(f'all:{self.val_allnum}')
            self.print(f'accuracy:{self.val_right / self.val_allnum}')
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")
            sys.stdout.flush()

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        # self.test_y_collection = []
        self.test_y_condition_H1nmr = []
        self.test_y_condition_C13nmr = []
        self.test_y_num_H_peak = []
        self.test_y_num_C_peak = []
        self.test_atomCount = []
        self.test_x = []
        self.test_e = []
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        '''*************'''
        # conditionAll = data.conditionVec
        # conditionAll = conditionAll.reshape(batch_length, self.vocabDim)
        condition_H1nmr = data.H_nmr
        condition_H1nmr = condition_H1nmr.reshape(batch_length, self.seq_len_H1, -1)

        condition_C13nmr = data.C_nmr
        condition_C13nmr = condition_C13nmr.reshape(batch_length, self.seq_len_C13)

        num_H_peak = data.num_H_peak
        num_C_peak = data.num_C_peak

        conditionVec = [condition_H1nmr, num_H_peak, condition_C13nmr, num_C_peak]

        input_X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        input_E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        input_y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        pred = self.forward(input_X, input_E, input_y, node_mask, conditionVec)
        #pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)

        # self.test_y_collection.append(data.conditionVec)
        self.test_y_condition_H1nmr.append(condition_H1nmr)
        self.test_y_condition_C13nmr.append(condition_C13nmr)
        self.test_y_num_H_peak.append(num_H_peak)
        self.test_y_num_C_peak.append(num_C_peak)
        self.test_atomCount.append(data.atom_count)
        self.test_x.append(X)
        self.test_e.append(E)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, condition=conditionVec, test= True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll :.4f}')

        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        # samples_left_to_save = self.cfg.general.final_model_samples_to_save
        # chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples, ident = [], 0

        # self.test_y_collection = torch.cat(self.test_y_collection, dim=0)
        self.test_atomCount = torch.cat(self.test_atomCount, dim=0)
        # self.test_y_collection = self.test_y_collection.view(-1, self.vocabDim)
        self.test_y_condition_H1nmr = torch.cat(self.test_y_condition_H1nmr, dim=0)
        self.test_y_condition_C13nmr = torch.cat(self.test_y_condition_C13nmr, dim=0)
        self.test_y_num_H_peak = torch.cat(self.test_y_num_H_peak, dim=0)
        self.test_y_num_C_peak = torch.cat(self.test_y_num_C_peak, dim=0)
        self.test_x = torch.cat(self.test_x, dim=0)
        self.test_e = torch.cat(self.test_e, dim=0)
        num_examples = self.test_y_condition_H1nmr.size(0)
        start_index = 0

        self.test_allnum = len(self.test_x)
        self.test_right = 0

        samples_left_to_generate = len(self.test_x)
        samples_left_to_save = len(self.test_x)
        chains_left_to_save = len(self.test_x)

        while samples_left_to_generate > 0:
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            if start_index + to_generate > num_examples:
                start_index = 0
            if to_generate > num_examples:
                ratio = to_generate // num_examples
                self.test_y_collection = self.test_y_collection.repeat(ratio + 1, 1)
                self.test_y_condition_H1nmr = self.test_y_condition_H1nmr.repeat(ratio + 1, 1)
                self.test_y_condition_C13nmr = self.test_y_condition_C13nmr.repeat(ratio + 1, 1)
                self.test_y_num_H_peak = self.test_y_num_H_peak.repeat(ratio + 1, 1)
                self.test_y_num_C_peak = self.test_y_num_C_peak.repeat(ratio + 1, 1)
                self.test_atomCount = self.test_atomCount.repeat(ratio + 1, 1)
                self.test_x = self.test_x.repeat(ratio + 1, 1)
                self.test_e = self.test_e.repeat(ratio + 1, 1)
                # num_examples = self.test_y_collection.size(0)
                num_examples = self.test_y_condition_H1nmr.size(0)

            '''**************'''
            # batch_y = self.test_y_collection[start_index:start_index + to_generate]
            batch_y_condition_H1nmr = self.test_y_condition_H1nmr[start_index:start_index + to_generate]
            batch_y_condition_C13nmr = self.test_y_condition_C13nmr[start_index:start_index + to_generate]
            batch_y_num_H_peak = self.test_y_num_H_peak[start_index:start_index + to_generate]
            batch_y_num_C_peak = self.test_y_num_C_peak[start_index:start_index + to_generate]
            batch_y = [batch_y_condition_H1nmr, batch_y_num_H_peak, batch_y_condition_C13nmr, batch_y_num_C_peak]

            '''_______________'''
            batch_atomCount = self.test_atomCount[start_index:start_index + to_generate]
            batch_x = self.test_x[start_index:start_index + to_generate]
            batch_e = self.test_e[start_index:start_index + to_generate]

            molecule_list, molecule_list_True = self.sample_batch(batch_id=ident,
                                                                  batch_size=to_generate,
                                                                  num_nodes=batch_atomCount,
                                                                  batch_condition=batch_y,
                                                                  save_final=to_save,
                                                                  keep_chain=chains_save,
                                                                  number_chain_steps=self.number_chain_steps,
                                                                  batch_X=batch_x,
                                                                  batch_E=batch_e)

            samples.extend(molecule_list)

            for i in range(to_generate):
                mol = self.mol_from_graphs(molecule_list[i][0].numpy(), molecule_list[i][1].numpy())
                mol_true = self.mol_from_graphs(molecule_list_True[i][0].numpy(), molecule_list_True[i][1].numpy())
                try:
                    fp1 = RDKFingerprint(mol)
                    fp2 = RDKFingerprint(mol_true)
                    # 计算Tanimoto相似度
                    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                    # 输出相似度
                    if similarity == 1:
                        self.test_right = self.test_right + 1
                except rdkit.Chem.KekulizeException:
                    print("Can't kekulize molecule")

            # samples.extend(self.sample_batch(batch_id=ident,
            #                                  batch_size=to_generate,
            #                                  num_nodes=batch_atomCount,
            #                                  batch_condition=batch_y,
            #                                  batch_x=batch_x,
            #                                  batch_e=batch_e,
            #                                  save_final=to_save,
            #                                  keep_chain=chains_save,
            #                                  number_chain_steps=self.number_chain_steps))
            ident += to_generate
            start_index += to_generate

            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
            sys.stdout.flush()
        self.print("Saving the generated graphs")
        self.print(f'right:{self.test_right}')
        self.print(f'all:{self.test_allnum}')
        self.print(f'accuracy:{self.test_right / self.test_allnum}')
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        self.print("Done testing.")


    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, conditionVec):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        input_X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        input_E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        input_y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        pred0 = self.forward(input_X, input_E, input_y, node_mask, conditionVec)
        #pred0 = self.forward(noisy_data, extra_data, node_mask, condition)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        # import numpy as np
        # t_int = torch.from_numpy(np.load("/tmp/t_int_fix.npy")).to(X.device)

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
        import numpy as np
        # np.savez("/home/liuxuwei01/PaddleMaterial/output/noisy_data.npz", **{k: (v.cpu().numpy() if hasattr(v, 'cpu') else (v.numpy() if hasattr(v, 'numpy') else v)) for k, v in noisy_data.items()})
        # noisy_data = np.load("/home/liuxuwei01/PaddleMaterial/output/noisy_data.npz")
        # noisy_data = {k: torch.from_numpy(v).to(X.device) for k, v in noisy_data.items()}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, condition, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask, condition)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch


        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, input_X, input_E, input_y, node_mask, conditionVec):
        # input_X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        # input_E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        # input_y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        '''***********'''
        # condition = torch.tensor(condition)
        '''____________'''
        # import numpy as np
        # input_X = torch.from_numpy(np.load("/home/liuxuwei01/PaddleMaterial/output/input_X.npy")).float().to(node_mask.device) # TODO
        # input_E = torch.from_numpy(np.load("/home/liuxuwei01/PaddleMaterial/output/input_E.npy")).float().to(node_mask.device)
        # input_y = torch.from_numpy(np.load("/home/liuxuwei01/PaddleMaterial/output/input_y.npy")).float().to(node_mask.device)
        return self.model(input_X, input_E, input_y, node_mask, conditionVec)




    def forward_sample(self, noisy_data, extra_data, node_mask, condition):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        '''************'''
        # condition = torch.tensor(condition)
        '''________________'''
        return self.model(X, E, y, node_mask, condition)

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_info.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol
    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, batch_condition, keep_chain: int, number_chain_steps: int,
                     save_final: int,  batch_X, batch_E, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        print(node_mask.shape)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, conditionVec=batch_condition,batch_X=batch_X, batch_E=batch_E)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        batch_X = torch.argmax(batch_X, dim=-1)
        batch_E = torch.argmax(batch_E, dim=-1)

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        molecule_list_True = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            atom_types_true = batch_X[i, :n].cpu()
            edge_types_true = batch_E[i, :n, :n].cpu()
            # print(f'atom_types{atom_types}')
            # print(f'atom_types_true{atom_types_true}')
            # print(f'edge_types{edge_types}')
            # print(f'edge_types_true{edge_types_true}')
            molecule_list.append([atom_types, edge_types])
            molecule_list_True.append([atom_types_true, edge_types_true])

        # Visualize chains
        if self.visualization_tools is not None:
            # self.print('Visualizing chains...')
            # current_path = os.getcwd()
            # num_molecules = chain_X.size(1)       # number of molecules
            # for i in range(num_molecules):
            #     result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
            #                                              f'epoch{self.current_epoch}/'
            #                                              f'chains/molecule_{batch_id + i}')
            #     if not os.path.exists(result_path):
            #         os.makedirs(result_path)
            #         _ = self.visualization_tools.visualize_chain(result_path,
            #                                                      chain_X[:, i, :].numpy(),
            #                                                      chain_E[:, i, :].numpy())
            #     self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            # self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            result_path_true = os.path.join(current_path,
                                            f'graphs/{self.name}/True_epoch{self.current_epoch}_b{batch_id}/')
            # self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.visualization_tools.visualizeNmr(result_path, result_path_true, molecule_list, molecule_list_True, save_final)
            self.print("Done.")

        return molecule_list, molecule_list_True

    # def sample_batch(self, batch_id: int, batch_size: int, batch_condition, batch_x, batch_e, keep_chain: int, number_chain_steps: int,
    #                  save_final: int, num_nodes=None):
    #     """
    #     :param batch_id: int
    #     :param batch_size: int
    #     :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
    #     :param save_final: int: number of predictions to save to file
    #     :param keep_chain: int: number of chains to save to file
    #     :param keep_chain_steps: number of timesteps to save for each chain
    #     :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
    #     """
    #     if num_nodes is None:
    #         n_nodes = self.node_dist.sample_n(batch_size, self.device)
    #     elif type(num_nodes) == int:
    #         n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
    #     else:
    #         assert isinstance(num_nodes, torch.Tensor)
    #         n_nodes = num_nodes
    #     n_max = torch.max(n_nodes).item()
    #     # Build the masks
    #     arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
    #     node_mask = arange < n_nodes.unsqueeze(1)
    #     print(node_mask.shape)
    #     # Sample noise  -- z has size (n_samples, n_nodes, n_features)
    #     z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
    #     X, E, y = z_T.X, z_T.E, z_T.y
    #
    #     assert (E == torch.transpose(E, 1, 2)).all()
    #     assert number_chain_steps < self.T
    #     chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
    #     chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))
    #
    #     chain_X = torch.zeros(chain_X_size)
    #     chain_E = torch.zeros(chain_E_size)
    #
    #     # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
    #     for s_int in reversed(range(0, self.T)):
    #         s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
    #         t_array = s_array + 1
    #         s_norm = s_array / self.T
    #         t_norm = t_array / self.T
    #
    #         # Sample z_s
    #         sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, conditionVec=batch_condition)
    #         X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
    #
    #         # Save the first keep_chain graphs
    #         write_index = (s_int * number_chain_steps) // self.T
    #         chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
    #         chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
    #
    #     # Sample
    #     sampled_s = sampled_s.mask(node_mask, collapse=True)
    #     X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
    #
    #
    #
    #     # Prepare the chain for saving
    #     if keep_chain > 0:
    #         final_X_chain = X[:keep_chain]
    #         final_E_chain = E[:keep_chain]
    #
    #         chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
    #         chain_E[0] = final_E_chain
    #
    #         chain_X = diffusion_utils.reverse_tensor(chain_X)
    #         chain_E = diffusion_utils.reverse_tensor(chain_E)
    #
    #         # Repeat last frame to see final sample better
    #         chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
    #         chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
    #         assert chain_X.size(0) == (number_chain_steps + 10)
    #
    #     molecule_list = []
    #     molecule_list_True = []
    #     for i in range(batch_size):
    #         n = n_nodes[i]
    #         atom_types = X[i, :n].cpu()
    #         edge_types = E[i, :n, :n].cpu()
    #         atom_types_true = batch_x[i, :n].cpu()
    #         edge_types_true = batch_e[i, :n, :n].cpu()
    #         molecule_list.append([atom_types, edge_types])
    #         molecule_list_True.append([atom_types_true, edge_types_true])
    #
    #     # Visualize chains
    #     if self.visualization_tools is not None:
    #         self.print('Visualizing chains...')
    #         current_path = os.getcwd()
    #         num_molecules = chain_X.size(1)       # number of molecules
    #         for i in range(num_molecules):
    #             result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
    #                                                      f'epoch{self.current_epoch}/'
    #                                                      f'chains/molecule_{batch_id + i}')
    #             if not os.path.exists(result_path):
    #                 os.makedirs(result_path)
    #                 _ = self.visualization_tools.visualize_chain(result_path,
    #                                                              chain_X[:, i, :].numpy(),
    #                                                              chain_E[:, i, :].numpy())
    #             self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
    #         self.print('\nVisualizing molecules...')
    #
    #         # Visualize the final molecules
    #         current_path = os.getcwd()
    #         result_path = os.path.join(current_path,
    #                                    f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
    #         result_path_true = os.path.join(current_path,
    #                                    f'graphs/{self.name}/True_epoch{self.current_epoch}_b{batch_id}/')
    #         self.visualization_tools.visualizeNmr(result_path, result_path_true,molecule_list,molecule_list_True, save_final)
    #         self.print("Done.")
    #
    #     return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, conditionVec, batch_X, batch_E):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward_sample(noisy_data, extra_data, node_mask, conditionVec)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

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
