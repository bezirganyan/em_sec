import subprocess
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments to compute mean & error bars."
    )
    # Number of times to run the training script
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    # Allow any additional arguments that should be passed to cnn.py
    args, extra_args = parser.parse_known_args()
    return args, extra_args

def main():
    args, extra_args = parse_args()

    # Remove any user-supplied wandb name (we will override it)
    filtered_args = []
    i = 0
    while i < len(extra_args):
        if extra_args[i].startswith("--wandb-name"):
            # If it's like '--wandb-name=value', skip it;
            # otherwise skip the next argument too (the value)
            if "=" in extra_args[i]:
                i += 1
            else:
                i += 2
            continue
        else:
            filtered_args.append(extra_args[i])
            i += 1

    # Create a base wandb name from all provided parameters (strip leading dashes)
    base_wandb_name = "_".join(arg.replace("--", "") for arg in filtered_args)
    # Base command to run the training script (assumes cnn.py is in the same directory)
    base_command = [sys.executable, "main.py"] + filtered_args

    for run in range(1, args.runs + 1):
        # Append run number to the wandb name
        run_wandb_name = f"{base_wandb_name}_run{run}"
        # Append the wandb name to the command line arguments
        cmd = base_command + ["--wandb-name", run_wandb_name]
        print(f"Running experiment {run} with command: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
