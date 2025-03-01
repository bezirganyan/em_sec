#!/usr/bin/env python3
import os
import subprocess

models = ['hyper', 'svp', 'ds']
datasets = ['cifar10', 'cifar100']
param = {'hyper': ('', []),
         'svp': ('beta', [1, 2, 3, 4, 5]),
         'ds': ('gamma', [0., 0.25, 0.5, 0.75, 1])}

# Template for the sbatch script with placeholders for substitution.
template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gpus=3090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/stdout_{job_name}.out
#SBATCH --error=slurm_logs/stderr_{job_name}.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate em_sec

srun -N1 -n1 python main.py --model {model} --epochs 300 --enable-wandb --dataset {dataset} --{param} {value}
"""

for model in models:
    for dataset in datasets:
        param_name, values = param[model]
        for value in values:
            job_name = f"{model}_{dataset}"
            script_content = template.format(job_name=job_name, model=model, dataset=dataset, param=param_name,
                                             value=value)
            script_filename = f"run_scripts/sbatch_{job_name}.sh"

            with open(script_filename, "w") as f:
                f.write(script_content)
            print(f"Created {script_filename}")

            result = subprocess.run(["sbatch", script_filename], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Submitted {script_filename}: {result.stdout.strip()}")
            else:
                print(f"Error submitting {script_filename}: {result.stderr.strip()}")
