
from typing import Tuple

import torch


def normalize_point_cloud(point_cloud: torch.Tensor) -> torch.Tensor:
    """Normalize point cloud to be centered at origin and have max vector norm of 1."""

    # find center (mean) of point cloud
    mu = torch.mean(point_cloud, axis=0)

    # find largest vector norm
    max_norm = torch.max(torch.linalg.norm(point_cloud, axis=1))

    # shift to be centered around origin and scale
    point_cloud_normalized = (point_cloud - mu) / max_norm
    return point_cloud_normalized


def add_noise(
    point_cloud: torch.Tensor,
    noise_type: str = "uniform",
    amount: float = 1e-2
) -> torch.Tensor:
    """Randomly perturbs clean point cloud with specified type of noise."""
    if noise_type == "uniform":
        random_noise = amount * torch.randn(size=point_cloud.shape)
    elif noise_type == "gaussian":
        random_noise = torch.normal(mean=0, std=amount, size=point_cloud.shape)
    else:
        random_noise = torch.zeros_like(point_cloud)
    return point_cloud + random_noise


def remove_points(point_cloud: torch.Tensor, num_remove: int) -> torch.Tensor:
    """Randomly removes a percentage of points in the point cloud uniformly."""
    num_points = point_cloud.shape[0]
    num_points_to_keep = max(0, num_points - num_remove)
    indices_to_keep = torch.randperm(n=num_points)[:num_points_to_keep]
    reduced_point_cloud = point_cloud[indices_to_keep]
    return reduced_point_cloud


def add_padding(point_cloud: torch.Tensor, max_points: int = 1024):
    """Adds zero-padding aka zero vectors according to the desired size."""
    num_pad = max_points - point_cloud.shape[0]
    padding = torch.zeros((num_pad, 3))
    padded_point_cloud = torch.cat([point_cloud, padding])
    return padded_point_cloud


def rotate_z_axis(point_cloud: torch.Tensor) -> torch.Tensor:
    """Applies a random z-axis rotation to the point cloud."""
    # randomly sample angle
    angle = torch.tensor(
        2 * torch.pi * torch.rand(1).item(), dtype=torch.float32, device=point_cloud.device
    )
    
    # z-axis rotation
    rot_z = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ], device=point_cloud.device)

    # apply rotation matrix
    rotated_point_cloud = point_cloud.matmul(rot_z.T)
    return rotated_point_cloud


def remove_neighboring_points(point_cloud: torch.Tensor, num_remove: int):
    """Removes a local section of points by remove the nearest neighbors starting 
    from an initial point."""
    starting_idx = torch.randint(0, point_cloud.shape[0], size=(1, ))
    # get the initial point
    center_point = point_cloud[starting_idx]
    
    # calculate pairwise distances between this point and all other points
    distances = torch.cdist(center_point, point_cloud)
    
    # find indices to keep (corresponding to n - num_remove farthest neighbors)
    _, indices_to_keep = torch.topk(
        distances, max(0, point_cloud.shape[0] - num_remove), largest=True
    )
    
    # remove points via indexing
    reduced_point_cloud = point_cloud[indices_to_keep].squeeze(0)
    return reduced_point_cloud


class RandomTransform:
    """Transformation that applies data augmentation for point completion/denoising tasks.

    Args:
        removal_amount: Percentage of points to remove.
            Default: 0.5.
        noise_amount: Additive scaling factor for noise.
            Default: 0.01.
        noise_type: Type of noise (i.e. uniform or gaussian)
        prob_both: Probability of applying both transformations (noise & removal)
            Default: 0.5.
        task: Learning task (i.e. completion or denoising). By default, both augmentations
            are applied, but specifying the task chooses one or the other only.
    """
    def __init__(
        self,
        removal_amount: float = 0.5,
        noise_amount: float = 1e-2,
        noise_type: str = "uniform",
        prob_both: float = 0.5,
        task: str = "None"
    ) -> None:
        self.removal_amount = removal_amount
        self.noise_amount = noise_amount
        self.noise_type = noise_type
        self.prob_both = prob_both
        self.task = task

    def __call__(self, point_cloud: torch.Tensor) -> Tuple:
        """Applies transformations to point cloud."""
        is_removed = is_noisy = False
        prob_both = 1.0 if self.task is not None else torch.rand(1)
        # apply both transformation
        if prob_both < self.prob_both:
            num_remove = int(point_cloud.shape[0] * self.removal_amount)
            point_cloud = remove_neighboring_points(point_cloud=point_cloud, num_remove=num_remove)
            point_cloud = add_noise(
                point_cloud=point_cloud, amount=self.noise_amount, noise_type=self.noise_type
            )
            is_removed = is_noisy = True
        else:
            if self.task is not None:
                prob_one = 1.0 if self.task == "completion" else 0
            else:
                prob_one = torch.rand(1)
            # apply only one transformation
            if prob_one < 0.5:
                point_cloud = add_noise(
                    point_cloud=point_cloud, amount=self.noise_amount, noise_type=self.noise_type)
                is_noisy = True
            else:
                num_remove = int(point_cloud.shape[0] * self.removal_amount)
                point_cloud = remove_neighboring_points(point_cloud=point_cloud, num_remove=num_remove)
                is_removed = True
                
        # also return a categorical label for the transformation
        # 0: identity, 1: is_removed, 2: is_noisy, 3: both
        if is_removed and is_noisy:
            transform_type = 3
        elif is_noisy:
            transform_type = 2
        elif is_removed:
            transform_type = 1
        else:
            transform_type = 0
        return transform_type, point_cloud