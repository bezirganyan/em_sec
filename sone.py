#!/usr/bin/env python3
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit a single SBATCH job with custom parameters."
    )
    parser.add_argument("--model", default="cnn", help="Model name (default: cnn)")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name (default: cifar10)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs (default: 300)")
    parser.add_argument("--job-name", help="Optional job name. If not provided, defaults to 'model_dataset_epochs'")

    # Capture extra parameters that are not defined
    args, extra_args = parser.parse_known_args()
    extra_params = " ".join(extra_args)

    # Determine job name based on input or defaults.
    job_name = args.job_name if args.job_name else f"{args.model}_{args.dataset}_{args.epochs}"

    # Create the SBATCH script content.
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/stdout_{job_name}.out
#SBATCH --error=slurm_logs/stderr_{job_name}.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate em_sec

srun -N1 -n1 python ../main.py --model {args.model} --epochs {args.epochs} --enable-wandb --dataset {args.dataset} {extra_params}
"""

    # Write the SBATCH script to a file.
    script_filename = f"sbatch_{job_name}.sh"
    with open(script_filename, "w") as f:
        f.write(sbatch_script)
    print(f"Created SBATCH script: {script_filename}")

    # Submit the SBATCH job.
    result = subprocess.run(["sbatch", script_filename], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Job {job_name} submitted successfully: {result.stdout.strip()}")
    else:
        print(f"Error submitting job {job_name}: {result.stderr.strip()}")

if __name__ == "__main__":
    main()
