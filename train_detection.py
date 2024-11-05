
import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.transform import RandomTransform
from dataset.shapenetcore import ShapeNetCore, ShapeNetCoreDefectDetection
from dataset.synthetic import SyntheticDefectData
from models.detection import Detector
from train import LRScheduler


NUM_GPUS = torch.cuda.device_count()
print(f"Number of GPUs available: {NUM_GPUS}")

# default hyperparameters
LR = 1e-5
PATIENCE = 20
WARMUP_RATIO = 0.1
BATCH_SIZE = 16 * NUM_GPUS or 1 # in case a GPU is not available
MAX_NUM_EPOCHS = 100
MAX_POINTS = 1024
NUM_LAYERS = 8
NUM_HEADS = 8
D_MODEL = 256
DROPOUT = 0.1
NUM_WORKERS = 8
DATASET_RATIO = 1


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
            x, y_true = batch

            # transfer data to gpu
            x, y_true = x.to(device), y_true.float().to(device)

            # run forward pass
            y_pred = model(x)

            # compute loss
            loss = nn.functional.binary_cross_entropy_with_logits(
                input=y_pred, target=y_true
            )
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
    num_correct = 0
    num_total = 0

    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, total=max_batches) as tq:
            tq.set_description(f"valid")
            for batch_idx, batch in enumerate(tq):
                
                # unpack batch
                x, y_true = batch

                # transfer data to gpu
                x, y_true = x.to(device), y_true.float().to(device)

                # run forward pass
                y_pred = model(x)

                # compute loss
                loss = nn.functional.binary_cross_entropy_with_logits(
                    input=y_pred, target=y_true
                )
                total_loss += loss.item()
                
                # count number of correct predictions
                y_pred_prob = nn.functional.sigmoid(y_pred)
                num_correct += (torch.where(y_pred_prob >= 0.5, 1, 0) == y_true).sum().item()
                num_total += x.shape[0]
                
                # display loss via logger
                mean_loss = total_loss / (batch_idx + 1)
                tq.set_postfix(
                    {"mean_loss": round(mean_loss, 7), "acc": round(num_correct / num_total, 6)}
                )

                if batch_idx == max_batches:
                    break
    mean_loss = total_loss / len(dataloader)
    print(f"Final accuracy: {num_correct / num_total}")
    return mean_loss


def train(args):
    "Runs training with the given config/args."
    
    # set master seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # set device
    device = torch.device(args.device)
    
    # create tensorboard logger
    tb_writer = SummaryWriter()

    # create datasets
    synthetic_defect_data = SyntheticDefectData(
        root="synthetic_data", 
        defect_type="removal" if args.task == "completion" else "noise"
    )
    num_synthetic_defects = len(synthetic_defect_data)
    
    # get nominal point clouds
    input_transform = RandomTransform(
        removal_amount=0.25, 
        noise_amount=0.05,
        task=args.task
    )

    # train on defects and nominals
    real_defect_data_train = ShapeNetCoreDefectDetection(
        root="Shapenetcore_benchmark",
        split="train",
        max_points=1365 if args.task == "completion" else 1024,
        input_transform=input_transform
    )
    real_nominal_data_train = ShapeNetCoreDefectDetection(
        root="Shapenetcore_benchmark",
        split="train",
        max_points=1024,
        input_transform=None
    )
    
    # aggregate train datasets
    real_defect_train_indices = torch.randperm(len(real_defect_data_train))[:num_synthetic_defects // 2]
    real_nominal_train_indices = torch.randperm(len(real_nominal_data_train))[:num_synthetic_defects // 2]
    
    real_defect_train_subset = Subset(real_defect_data_train, indices=real_defect_train_indices)
    real_nominal_train_subset = Subset(real_nominal_data_train, indices=real_nominal_train_indices)
    train_data = ConcatDataset([real_defect_train_subset, real_nominal_train_subset, synthetic_defect_data])
    
    # test on defects and nominals
    real_defect_data_test = ShapeNetCoreDefectDetection(
        root="Shapenetcore_benchmark",
        split="test",
        max_points=1365 if args.task == "completion" else 1024,
        input_transform=input_transform
    )
    real_nominal_data_test = ShapeNetCoreDefectDetection(
        root="Shapenetcore_benchmark",
        split="test",
        max_points=1365 if args.task == "completion" else 1024,
        input_transform=None
    )
    
    # aggregate test datasets
    real_defect_test_indices = torch.randperm(len(real_defect_data_test))
    real_nominal_test_indices = torch.randperm(len(real_nominal_data_test))
    
    real_defect_test_subset = Subset(real_defect_data_train, indices=real_defect_test_indices)
    real_nominal_test_subset = Subset(real_nominal_data_train, indices=real_nominal_test_indices)
    test_data = ConcatDataset([real_defect_test_subset, real_nominal_test_subset])
    
    print(
        f"Total number of examples: {len(train_data)}, synthetic: {num_synthetic_defects}, "
        f"real: {len(train_data) - num_synthetic_defects}"
    )
    
    # get train-val split
    train_data, val_data = random_split(train_data, [0.5, 0.5])
    
    # wrap datasets into data loaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # instantiate model and push to device
    model = Detector(
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
    
    print(f"Starting training...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience = args.patience
    count = 0
    
    # main training loop
    for epoch in range(args.max_num_epochs):
        print(f"Epoch [{epoch + 1}/{args.max_num_epochs}]")

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # reset early stopping counter
            count = 0
            
            # make checkpoint directory
            if not Path(args.root, "checkpoints").exists():
                Path(args.root, "checkpoints").mkdir(parents=True)

            # save best model
            model_name = model.__class__.__name__ if NUM_GPUS <= 1 else model.module.__class__.__name__
            torch.save(
                obj={
                    "model": model.cpu().state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "warmup_scheduler": scheduler.warmup_scheduler.state_dict(),
                    "decay_scheduler": scheduler.lr_decay_scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "epoch": epoch,
                    **args.__dict__
                }, 
                f=f"{args.root}/checkpoints/{model_name}_{args.task}.pth"
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
    
    # evaluate model on test data
    state_dict = torch.load(f"{args.root}/checkpoints/{model_name}_{args.task}.pth")
    model.load_state_dict(state_dict["model"])
    validate_epoch(model=model, dataloader=test_loader)
        
            
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
    parser.add_argument('--task',  type=str, default="completion", help="Learning task i.e. (completion or denoising) for detection.")

    # parse args
    args = parser.parse_args()
    
    # run training
    print("Training configuration:\n" + "-" * 85)
    for key, val in args.__dict__.items():
        print(f"{key}: {val}")
    print("-" * 85)
    train(args)
    

if __name__ == "__main__":     
    main()
    