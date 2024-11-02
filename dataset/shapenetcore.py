import zipfile
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from .transform import add_padding, normalize_point_cloud, remove_points, rotate_z_axis


def extract_data(local_dir: str) -> None:
    """Download & extract a tiny version of ShapeNetCore locally for testing."""
    # create new directory (if necessary)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # skip download if dataset already found
    if Path(local_dir, "Shapenetcore_benchmark").exists():
        print(f"Dataset already downloaded and found at {local_dir}.")
        return
    
    print(f"Downloading dataset to {local_dir}")
    # download shapenetcore from my huggingface account
    zip_path = hf_hub_download(
        repo_id="kianzohoury/shapenetcore",
        filename="archive.zip",
        repo_type="dataset",
        local_dir=local_dir
    )
    
    # extract from the downloaded zip file
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, mode='r') as file:
        file.extractall(local_dir)
    print(f"Files extracted to {local_dir}")
        
    # remove original zip file
    Path(zip_path).unlink()
    

class ShapeNetCore(Dataset):
    """ShapeNetCore dataset with applied augmentions for denoising & point completion.

    Args:
        root: Root directory of the dataset path.
        split: Dataset split (i.e. train, val, test).
            Default: "train".
        max_points: Max size of a point cloud.
            Default: 2048.
        downsampling_mode: Method of downsampling point clouds.
            Default: "uniform".
        input_transform: Transformation to apply to input point clouds.
            Default: None.
        use_rotations: Apply z-axis rotations.
            Default: True.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        max_points: int = 2048,
        downsampling_mode: str = "uniform",
        input_transform = None,
        use_rotations: bool = True
    ) -> None:
        super(ShapeNetCore, self).__init__()
        self.root_dir = root
        self.split = split
        self.max_points = max_points
        self.downsampling_mode = downsampling_mode
        self.input_transform = input_transform
        self.use_rotations = use_rotations
        self.metadata = []

        # load metadata file
        with open(f"{self.root_dir}/{self.split}_split.json", mode="r") as f:
            self.metadata = json.load(f)

        # map class labels to dataset indices
        self.class_label_to_idx = defaultdict(list)
        for idx in range(len(self.metadata)):
            class_label = self.metadata[idx][1]
            self.class_label_to_idx[class_label].append(idx)
            
        # transformation/augmentation type
        self.id_to_defect_type = {
            0: "nominal",
            1: "removal",
            2: "noise",
            3: "removal and noise"
        }

    def __len__(self) -> int:
        """Returns length of dataset."""
        return len(self.metadata)

    def __getitem__(self, idx) -> Tuple:
        """Retrieves an example from the dataset.

        Specifically, it returns a tuple containing the class index,
        string label, transformation type, followed by the noisy (x) and
        clean (y) point cloud tensors.
        """
        class_idx, class_label, npy_path = self.metadata[idx]
        # load point cloud as numpy array
        target_point_cloud = np.load(Path(self.root_dir, npy_path))

        # swap coordinate positions because they come as xzy
        target_point_cloud = target_point_cloud[:, [0, 2, 1]]
        num_points = target_point_cloud.shape[0]
        
        # convert to tensor
        target_point_cloud = torch.tensor(target_point_cloud, dtype=torch.float32)

        # normalize point cloud
        target_point_cloud = normalize_point_cloud(target_point_cloud)
        
        # apply z-axis rotation
        target_point_cloud = rotate_z_axis(target_point_cloud)

        # downsample to max_points if necessary
        if num_points > self.max_points:
            # use uniform random sampling (quick & dirty)
            if self.downsampling_mode == "uniform":
                num_remove = max(0, num_points - self.max_points)
                target_point_cloud = remove_points(
                    point_cloud=target_point_cloud, num_remove=num_remove
                )
            # could be used for different down-sampling techniques (e.g. farthest point)
            else:
                pass
            
        # zero-pad point clouds with zero vectors if necessary    
        elif num_points < self.max_points:
            target_point_cloud = add_padding(
                point_cloud=target_point_cloud, max_points=self.max_points
            ) 

        # create the noisy point cloud (x) by applying transformations on target (y)
        if self.input_transform is not None:
            transform_type, noisy_point_cloud = self.input_transform(
                target_point_cloud.clone() # copying to be extra careful
            )
        else:
            noisy_point_cloud = target_point_cloud.clone()
            transform_type = 0        
        return class_idx, class_label, transform_type, noisy_point_cloud, target_point_cloud
    

if __name__ == "__main__":
    # simply download to current directory
    extract_data("./")