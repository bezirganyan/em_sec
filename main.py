# define the main file for training the model in cnn.py from cifar10 dataset from datasets.py using pytorch_lightning
import argparse
import wandb

from dataset import CIFAR10DataModule
from models.beta import CIFAR10BettaModel
from models.cnn import CIFAR10Model

import pytorch_lightning as pl
import torch

from models.dir_beta import CIFAR10HyperModel
from models.ds_baseline import CIFAR10DSModel
from models.enn import CIFAR10EnnModel
from models.svp_baseline import CIFAR10SVPModel
from utils import OnKeyboardInterruptCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_interval_steps', type=int, default=100)
    parser.add_argument('--tensorboard_path', type=str, default='lightning_logs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--enable-wandb', action='store_true', default=False)
    parser.add_argument('--unc-calib', action='store_true', default=False)

    return parser.parse_args()

def main():
    args = parse_args()

    wandb_mode = 'online' if args.enable_wandb else 'disabled'
    wandb_name = f"{args.model}_cifar10_{args.beta}_beta_{args.epochs}_epochs_{args.learning_rate}_lr"
    wandb.init(project='VagueFusion', name=wandb_name, config=vars(args), mode=wandb_mode)

    data = CIFAR10DataModule(batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir)
    data.setup()
    if args.model == 'cnn':
        model = CIFAR10Model(num_classes=args.num_classes, learning_rate=args.learning_rate)
    elif args.model == 'enn':
        model = CIFAR10EnnModel(num_classes=args.num_classes, learning_rate=args.learning_rate, uncertainty_calibration=args.unc_calib)
    elif args.model == 'beta':
        model = CIFAR10BettaModel(num_classes=args.num_classes, learning_rate=args.learning_rate)
    elif args.model == 'hyper':
        model = CIFAR10HyperModel(num_classes=args.num_classes, learning_rate=args.learning_rate, beta=args.beta)
    elif args.model == 'ds':
        model = CIFAR10DSModel(num_classes=args.num_classes, learning_rate=args.learning_rate)
    elif args.model == 'svp':
        model = CIFAR10SVPModel(num_classes=args.num_classes, learning_rate=args.learning_rate, beta=args.beta)
    else:
        raise ValueError(f"Model {args.model} not supported")

    trainer = pl.Trainer(
        accelerator="auto",
        log_every_n_steps=args.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(args.tensorboard_path, "pipeline"),
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[OnKeyboardInterruptCallback()]
    )
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()
    print("\nStarting full training...\n")
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Testing the model")
    results = trainer.test(model, test_dataloader)[0]
    print(f'Test results: {results}')

if __name__ == '__main__':
    main()