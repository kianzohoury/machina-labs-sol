
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.shapenetcore import ShapeNetCore
from dataset.transform import RandomTransform
from models.completion import CompletionTransformer
from models.denoiser import DenoiserTransformer
from train import get_baseline_chamfer_dist, set_seed, validate_epoch, worker_init_fn

def test_models(
    model_dir: str, save_path: str = "./test_results.csv", batch_size: int = 8,
) -> None:
    """Tests model checkpoints given a model directory on hold-out test data and saves results."""
    df_col_labels = [
        "model",
        "noise_amount",
        "removal_ratio",
        "avg_chamf_dist",
        "avg_chamf_dist_baseline"
    ]
    df_rows = []
    
    # set validation seed for fixed data augmentations
    set_seed(0)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = list(Path(model_dir).glob("*.pth"))
    for checkpoint_name in model_paths:
        # load checkpoint
        state_dict = torch.load(
            f=checkpoint_name, map_location=device
        )

        # create data transform
        input_transform = RandomTransform(
            removal_amount=state_dict["removal_amount"],
            noise_amount=state_dict["noise_amount"],
            noise_type=state_dict["noise_type"],
            task=state_dict["task"]
        )

        # create datasets
        test_data = ShapeNetCore(
            root="Shapenetcore_benchmark",
            split="test",
            max_points=state_dict["max_points"],
            input_transform=input_transform,
            use_rotations=state_dict["use_rotations"]
        )   
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        # get baseline mean chamfer distance score
        baseline_score = get_baseline_chamfer_dist(test_loader, device=device)

        # instantiate model and push to device
        if state_dict["task"] == "completion":
            model = CompletionTransformer(
                num_layers=state_dict["num_layers"],
                num_heads=state_dict["num_heads"],
                d_model=state_dict["d_model"],
                dropout=state_dict["dropout"],
                num_queries=int(state_dict["max_points"] * state_dict["removal_amount"])
            ).to(device)
        elif state_dict["task"] == "denoising":
            model = DenoiserTransformer(
                num_layers=state_dict["num_layers"],
                num_heads=state_dict["num_heads"],
                d_model=state_dict["d_model"],
                dropout=state_dict["dropout"],
            ).to(device)
        else:
            # ignore other possible model tasks (like combining both tasks) for now
            return 

        # necessary if distributed training was used
        if all(key.startswith("module.") for key in state_dict["model"].keys()):
            model = nn.DataParallel(model)

        # load weights
        model.load_state_dict(state_dict["model"])
        print(f"Testing model: {Path(checkpoint_name).stem}")
        test_loss = validate_epoch(
            model=model,
            dataloader=test_loader
        )
        avg_chamfer_dist = test_loss
        avg_baseline_score = baseline_score
        df_rows.append([
            Path(checkpoint_name).stem,
            state_dict["noise_amount"],
            state_dict["removal_amount"],
            avg_chamfer_dist,
            avg_baseline_score
        ])
        print(
            f"Average Chamfer Distance: {avg_chamfer_dist}, "
            f"Baseline: {avg_baseline_score}"
        )

    # create dataframe and save results to .csv file
    pd.DataFrame(df_rows, columns=df_col_labels).to_csv(save_path, index=False)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing configuration")

    # add default args
    parser.add_argument('--model_dir', type=str, default=Path.cwd() , help="Directory where models are stored.")
    parser.add_argument('--save_path', type=str, default="test_results.csv", help="Path to save test metrics to.")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to run inference on.')
    
    # parse args
    args = parser.parse_args()
    
    # run testing
    test_models(**args.__dict__)
    