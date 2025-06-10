#!/bin/bash
#SBATCH --job-name=Graph_DiT
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_debug
#SBATCH --nodelist=gpu[09]
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=300G
#SBATCH --output=./%j.out
#SBATCH --error=./%j.error
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue

#python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 main.py --config-name=config_new.yaml model.ensure_connected=True dataset.task_name='tokenized_input' dataset.guidance_target='tokenized_input'
python main.py
#python main.py --config-name=config.yaml model.ensure_connecte
# d=True dataset.task_name='O2-N2-CO2' dataset.guidance_target='O2-N2-CO2'