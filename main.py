import argparse
import wandb

from dataset import CIFAR100DataModule, CIFAR10DataModule, RxRx1DataModule
from models.beta import BetaModel
from models.cnn import StandardModel

import pytorch_lightning as pl
import torch

from models.conv_models import BasicBlock, ResNet
from models.dir_beta import EMSECModel
from models.ds_baseline import DSModel
from models.enn import ENNModel
from models.linear_models import DenseClassifier
from models.svp_baseline import SVPModel
from utils import OnKeyboardInterruptCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_interval_steps', type=int, default=100)
    parser.add_argument('--tensorboard_path', type=str, default='lightning_logs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--ann_start', type=int, default=100)
    parser.add_argument('--ann_end', type=int, default=300)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--toli', type=int, default=2)
    parser.add_argument('--enable-wandb', action='store_true', default=False)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--unc-calib', action='store_true', default=False)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--project', type=str, default='VagueFusion')

    return parser.parse_args()

def wandb_name_from_args(args):
    if args.model == 'cnn':
        return f"{args.model}_{args.dataset}_{args.epochs}_epochs"
    elif args.model == 'enn':
        return f"{args.model}_{args.dataset}_{args.epochs}_epochs_{args.unc_calib}_uncalib"
    elif (args.model == 'hyper') or (args.model == 'svp'):
        return f"{args.model}_{args.dataset}_{args.beta}_beta_{args.epochs}_epochs"
    elif args.model == 'ds':
        return f"{args.model}_{args.dataset}_{args.gamma}_gamma_{args.epochs}_epochs"
    else:
        return f"{args.model}_{args.dataset}_{args.epochs}_epochs"


def main():
    args = parse_args()

    wandb_mode = 'online' if args.enable_wandb else 'disabled'
    wandb_name = wandb_name_from_args(args) if args.wandb_name is None else args.wandb_name

    if args.dataset == 'cifar10':
        data = CIFAR10DataModule(batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir)
        num_classes = 10
        model_backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif args.dataset == 'cifar100':
        data = CIFAR100DataModule(batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir)
        num_classes = 100
        model_backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif args.dataset == 'rxrx1':
        data = RxRx1DataModule(batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir)
        num_classes = 1139
        model_backbone = DenseClassifier(in_features=128, out_features=num_classes, hidden_features=(256, 256))
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    data.setup()

    if args.model == 'cnn':
        model = StandardModel.load_from_checkpoint(args.ckpt_path) if args.ckpt_path else \
            StandardModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate)
    elif args.model == 'enn':
        model = ENNModel.load_from_checkpoint(args.ckpt_path) if args.ckpt_path else \
            ENNModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate, uncertainty_calibration=args.unc_calib)
    elif args.model == 'beta':
        model = BetaModel.load_from_checkpoint(args.ckpt_path) if args.ckpt_path else \
            BetaModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate)
    elif args.model == 'hyper':
        model = EMSECModel.load_from_checkpoint(args.ckpt_path, strict=False) if args.ckpt_path else \
            EMSECModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate, beta=args.beta,
                       annealing_start=args.ann_start, annealing_end=args.ann_end)
    elif args.model == 'ds':
        model = DSModel.load_from_checkpoint(args.ckpt_path) if args.ckpt_path else \
            DSModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate, nu=args.gamma, tol_i=args.toli)
    elif args.model == 'svp':
        model = SVPModel.load_from_checkpoint(args.ckpt_path) if args.ckpt_path else \
            SVPModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate, beta=args.beta)
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
    log_path = trainer.logger.log_dir
    print(f"Logging to {log_path}")
    args.log_path = log_path
    wandb.init(project=args.project, name=wandb_name, config=vars(args), mode=wandb_mode)

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()
    print("\nStarting full training...\n")

    if not args.test:
        try:
            trainer.fit(model, train_dataloader, val_dataloader)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Testing the model")
    if args.model == 'svp':
        results = trainer.test(model, test_dataloader)[0]
        print(f'Test results for main beta {args.beta}: {results}')
        trained_state = model.state_dict()
        for beta_val in [1, 2, 3, 4, 5, 10]:
            if beta_val == args.beta:
                continue
            wandb.finish()
            test_wandb_name = wandb_name.replace(f'beta_{args.beta}', f'beta_{beta_val}')
            wandb.init(project=args.project, name=test_wandb_name, config=vars(args), mode=wandb_mode)
            model_backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            new_model = SVPModel(model_backbone, num_classes=num_classes, learning_rate=args.learning_rate, beta=beta_val)
            new_model.load_state_dict(trained_state)
            new_model.set_params = {
                "c": num_classes,
                "svptype": "fb",
                "beta": beta_val
            }
            new_model.beta_param = beta_val
            results = trainer.test(new_model, test_dataloader)[0]
            print(f'Test results for beta {beta_val}: {results}')
            wandb.finish()
    else:
        results = trainer.test(model, test_dataloader)[0]
        print(f'Test results: {results}')

if __name__ == '__main__':
    main()
