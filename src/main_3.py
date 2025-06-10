import os
import pathlib
import warnings

import sys
sys.path.insert(0, '/home/liuxuwei01/molecular2molecular/src')

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from DDDM import DiscreteDenoisingDiffusionMolecular_condition
from diffusion_model_discrete_condition import DiscreteDenoisingDiffusionCondition
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures

from metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete

from analysis.visualization import MolecularVisualization

from datasets import CHnmr_dataset
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance."
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future."
)



import torch
import numpy as np
import random


torch._C._set_warnAlways(False)


def set_seed(seed):
    random.seed(seed)                  # Python 内置随机
    np.random.seed(seed)               # NumPy 随机
    torch.manual_seed(seed)            # CPU 生成的随机数
    torch.cuda.manual_seed(seed)       # GPU 单卡
    torch.cuda.manual_seed_all(seed)   # GPU 多卡
    torch.backends.cudnn.deterministic = True   # 强制 cudnn 使用确定性算法
    torch.backends.cudnn.benchmark = False      # 禁止 benchmark，确保每次算法一致

def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

    model = DiscreteDenoisingDiffusionMolecular_condition.load_from_checkpoint(resume, **model_kwargs)

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = DiscreteDenoisingDiffusionMolecular_condition.load_from_checkpoint(resume_path, **model_kwargs)

    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'
    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    datamodule = CHnmr_dataset.CHnmrDataModule(cfg)
    dataset_infos = CHnmr_dataset.CHnmrinfos(datamodule=datamodule, cfg=cfg)
    train_smiles = CHnmr_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features, conditionDim=cfg.model.conditdim)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos,
                    'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics,
                    'visualization_tools': visualization_tools,
                    'extra_features': extra_features,
                    'domain_features': domain_features}


    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    model = DiscreteDenoisingDiffusionCondition(cfg=cfg, **model_kwargs)
    #torch.save(model.state_dict(),'/home/liuxuwei01/PaddleMaterial/output/init_lxw_step1.pth')

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=3,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="auto",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=1,
                      logger = [])

    if not cfg.general.test_only:
        set_seed(55)
        # trainer.fit(model, datamodule=datamodule, ckpt_path='/home/liuxuwei01/PaddleMaterial/output/step4_best.ckpt')
        # trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['test']:
            # trainer.test(model, datamodule=datamodule)
            trainer.test(model, datamodule=datamodule,ckpt_path='/public/home/szlab_wubinglan/DeNMR/pretrained/step4_best.ckpt')
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()



