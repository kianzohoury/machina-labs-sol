
import argparse
import itertools
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset.shapenetcore import extract_data, ShapeNetCore
from dataset.transform import RandomTransform
from kaolin.metrics.pointcloud import chamfer_distance
from models.completion import CompletionTransformer
from models.denoiser import DenoiserTransformer
from models.conv import DenoiserConv


NUM_GPUS = torch.cuda.device_count()
print(f"Number of GPUs available: {NUM_GPUS}")

# default hyperparameters
LR = 1e-4
PATIENCE = 10
WARMUP_RATIO = 0.1
BATCH_SIZE = 16 * NUM_GPUS or 1 # in case a GPU is not available
MAX_NUM_EPOCHS = 100
MAX_POINTS = 1024
NUM_LAYERS = 8
NUM_HEADS = 8
D_MODEL = 256
DROPOUT = 0.1
NUM_WORKERS = 8
DATASET_RATIO = 0.1

# training experiments: four fixed combinations representing the strength of point cloud augmentations
# change these if you'd like to experiment with stronger augmentations
NOISE_AMOUNTS = [0.05, 0.075]
REMOVAL_AMOUNTS = [0.25, 0.5]
NOISE_TYPE = "gaussian"
USE_ROTATIONS = True


class LRScheduler:
    """Custom learning rate scheduler that initially applies linear scaling for warmup and 
    cosine annealing (decay) after the warmup is finished."""
    def __init__(
        self,
        optimizer: Any,
        step_size: int,
        total_steps: int,
        warmup_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.num_steps = 0

        # define warmup scheduler
        self.warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        # define decay scheduler
        self.lr_decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_steps - self.warmup_steps
        )

    def step(self):
        """Updates learning rate."""
        self.num_steps += self.step_size
        if self.num_steps <= self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.lr_decay_scheduler.step()
            
    def get_current_lr(self) -> float:
        """Gets the current learning rate."""
        if self.num_steps <= self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()[0]
        else:
            return self.lr_decay_scheduler.get_last_lr()[0]


def worker_init_fn(worker_id) -> None:
    """Custom initialization function that helps fix data augmentation behavior per worker."""
    seed = torch.initial_seed() % (2 ** 32)
    torch.manual_seed(seed + worker_id)
    
    
def set_seed(seed: int) -> None:
    """Sets random state given a seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    
def train_epoch(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    dataloader: DataLoader,
    max_batches = None
) -> float:
    """Trains model over the entire training set and returns the mean loss."""
    max_batches = len(dataloader) if max_batches is None else max_batches
    max_batches = min(len(dataloader), max_batches)
    device = list(model.parameters())[0].device
    total_loss = 0

    model.train()
    with tqdm(dataloader, total=max_batches) as tq:
        tq.set_description(f"train")
        for batch_idx, batch in enumerate(tq):

            # unpack batch
            class_idx, class_label, transform_type, x, y_true = batch

            # transfer data to gpu
            x, y_true = x.to(device), y_true.to(device)

            # run forward pass
            y_pred = model(x)
            
            # compute loss
            loss = chamfer_distance(p1=y_pred, p2=y_true).mean()
            total_loss += loss.item()

            # backprop + update parameters + update lr
            loss.backward()
            optimizer.step()
            scheduler.step()

            # clear gradients
            optimizer.zero_grad()

            # display loss via logger
            mean_loss = total_loss / (batch_idx + 1)
            tq.set_postfix(
                {"mean_loss": round(mean_loss, 7), "lr": scheduler.get_current_lr()}
            )

            if batch_idx == max_batches:
                break
    mean_loss = total_loss / len(dataloader)
    return mean_loss


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches = None
) -> float:
    """Validates model over the entire validation set and returns the mean loss."""
    max_batches = len(dataloader) if max_batches is None else max_batches
    max_batches = min(len(dataloader), max_batches)
    device = list(model.parameters())[0].device
    total_loss = 0

    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, total=max_batches) as tq:
            tq.set_description(f"valid")
            for batch_idx, batch in enumerate(tq):

                # unpack batch
                class_idx, class_label, transform_type, x, y_true = batch

                # transfer data to gpu
                x, y_true = x.to(device), y_true.to(device)

                # run forward pass
                y_pred = model(x)
                
                # compute loss
                loss = chamfer_distance(p1=y_pred, p2=y_true).mean()
                total_loss += loss.item()

                # display loss via logger
                mean_loss = total_loss / (batch_idx + 1)
                tq.set_postfix({"mean_loss": round(mean_loss, 7)})

                if batch_idx == max_batches:
                    break
    mean_loss = total_loss / len(dataloader)
    return mean_loss


def train(args):
    "Runs training with the given config/args."
    
    # set master seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    
    if args.task == "completion":
        augmentation_combos = list(set(itertools.product([0, 0], REMOVAL_AMOUNTS)))
    elif args.task == "denoising":
        augmentation_combos = list(set(itertools.product(NOISE_AMOUNTS, [0, 0])))
    else:
        augmentation_combos = list(itertools.product(NOISE_AMOUNTS, REMOVAL_AMOUNTS))
    
    print("Training experiments that will be run (% noise, % missing):")
    for i in range(len(augmentation_combos)):
        print(f"Experiment {i + 1}: {augmentation_combos[i][0], augmentation_combos[i][1] * 100}")
    print()
        
    # this allows each model to receive the same training augmentations
    training_augmentation_seeds = torch.randint(2, 2**32 - 1, size=(MAX_NUM_EPOCHS, ))

    # set device
    device = torch.device(args.device)
    
    # create tensorboard logger
    tb_writer = SummaryWriter()
    
    # run experiments
    for experiment_idx in range(len(augmentation_combos)):
        noise_amount, removal_amount = augmentation_combos[experiment_idx]
        
        # create data transform
        input_transform = RandomTransform(
            removal_amount=removal_amount,
            noise_amount=noise_amount,
            noise_type=args.noise_type,
            task=args.task
        )

        # create datasets
        train_data = ShapeNetCore(
            root="Shapenetcore_benchmark",
            split="train",
            max_points=args.max_points,
            input_transform=input_transform,
            use_rotations=args.use_rotations
        )   
        val_data = ShapeNetCore(
            root="Shapenetcore_benchmark",
            split="val",
            max_points=args.max_points,
            input_transform=input_transform,
            use_rotations=args.use_rotations
        )

        # wrap datasets into data loaders
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        # get baseline score to beat
        baseline_chamf_dist = get_baseline_chamfer_dist(
            dataloader=val_loader, device=device
        )

        # instantiate model and push to device
        if args.task == "completion":
            model = CompletionTransformer(
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                d_model=args.d_model,
                dropout=args.dropout,
                num_queries=int(args.max_points * removal_amount)
            ).to(device)
        else:
            # 1D convolutional model as a baseline
            if args.conv:
                model = DenoiserConv(d_model=args.d_model).to(device)
            else:
                model = DenoiserTransformer(
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    d_model=args.d_model,
                    dropout=args.dropout,
                ).to(device)
                
        # for distributed training
        if NUM_GPUS > 1:
            print("Running distributed training.")
            model = nn.DataParallel(model)

        # instantiate optimizer and LR warmup scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr)
        max_batches = round((len(train_data) / args.batch_size) * args.dataset_ratio)
        total_steps = max_batches * args.max_num_epochs
        scheduler = LRScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            step_size=1,
            warmup_ratio=args.warmup_ratio
        )

        print(f"Experiment: [{experiment_idx + 1}/{len(augmentation_combos)}]")
        print(f"eps={noise_amount}, r={removal_amount}\n" + "-" * 85)
        print(f"Starting training...")
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        patience = args.patience
        count = 0
        
        # main training loop
        for epoch in range(args.max_num_epochs):
            print(f"Epoch [{epoch + 1}/{args.max_num_epochs}]")

            # set training augmentation seed
            set_seed(training_augmentation_seeds[epoch])

            # run training loop
            train_loss = train_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                max_batches=max_batches
            )
            # record training loss
            train_losses.append(train_loss)

            # set validation seed for fixed data augmentations
            set_seed(1)

            # run validation loop
            val_loss = validate_epoch(
                model=model,
                dataloader=val_loader,
                max_batches=max_batches
            )
            # record validation loss
            val_losses.append(val_loss)
            
            # log to tensorboard
            tb_writer.add_scalar("loss/train", scalar_value=train_loss, global_step=epoch)
            tb_writer.add_scalar("loss/val", scalar_value=val_loss, global_step=epoch)

            # score to beat/for sanity reasons
            print(f"Baseline Chamfer Distance: {baseline_chamf_dist}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # reset early stopping counter
                count = 0
                
                # make checkpoint directory
                if not Path(args.root, "checkpoints").exists():
                    Path(args.root, "checkpoints").mkdir(parents=True)

                # save best model
                torch.save(
                    obj={
                        "model": model.cpu().state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # "warmup_scheduler": scheduler.warmup_scheduler.state_dict(),
                        # "decay_scheduler": scheduler.lr_decay_scheduler.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "epoch": epoch,
                        "noise_amount": noise_amount,
                        "removal_amount": removal_amount,
                        **args.__dict__
                    }, 
                    f=f"{args.root}/checkpoints/{model.__class__.__name__}_{experiment_idx + 1}.pth"
                )
                
                print("Best model saved.")
                # push model back to device
                model = model.to(device)
            else:
                # increment early stopping counter
                count += 1
                if count == patience:
                    # stop training
                    print("Stopping training early.")
                    break
        print("Training complete.\n")
        
        
def get_baseline_chamfer_dist(dataloader: DataLoader, device: str) -> float:
    """Gets the baseline between augmented inputs and clean labels 
    (average Chamfer Distance across the whole dataset).
    """
    total_chamf_dist_without_model = 0
    for example in dataloader:
        noisy_point_cloud, target_point_cloud = example[-2:]
        chamf_dist = chamfer_distance(
            p1=noisy_point_cloud.to(device),
            p2=target_point_cloud.to(device)
        )
        total_chamf_dist_without_model += chamf_dist.mean().item()
    mean_chamf_dist_without_model = total_chamf_dist_without_model / len(dataloader)
    return mean_chamf_dist_without_model

    
def main():
    """Main function using CLI."""
    
    parser = argparse.ArgumentParser(description="Training configuration")

    # add default args
    parser.add_argument('--root', type=str, default=Path.cwd() , help="Root directory to store dataset and model artifacts")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--patience', type=float, default=PATIENCE, help='Number of epochs with no improvement before stopping early.')
    parser.add_argument('--warmup_ratio', type=float, default=WARMUP_RATIO, help='Warmup ratio')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--max_num_epochs', type=int, default=MAX_NUM_EPOCHS, help='Maximum number of epochs')
    parser.add_argument('--max_points', type=int, default=MAX_POINTS, help='Maximum number of points')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--d_model', type=int, default=D_MODEL, help='Dimensionality of the K, Q, and V matrices for self-attention')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
    parser.add_argument('--dataset_ratio', type=float, default=DATASET_RATIO, help='Ratio of the training set to use')
    parser.add_argument('--noise_type', type=str, default=NOISE_TYPE, help="Type of noise to use (i.e. uniform or gaussian)")
    parser.add_argument('--use_rotations',  type=bool, default=USE_ROTATIONS, help="Applies random z-axis rotations.")
    parser.add_argument('--task',  type=str, default="completion", help="Learning task i.e. (completion or denoising) to run.")
    parser.add_argument('--conv',  action="store_true", help="Runs the baseline convolutional model.")

    # parse args
    args = parser.parse_args()
    
    # download dataset
    extract_data(local_dir=args.root)
    
    # run training
    print("Training configuration:\n" + "-" * 85)
    for key, val in args.__dict__.items():
        print(f"{key}: {val}")
    print("-" * 85)
    train(args)
    

if __name__ == "__main__":     
    main()
    