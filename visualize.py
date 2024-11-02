

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset.shapenetcore import ShapeNetCore

def visualize_point_cloud_comparison(
    dataset: ShapeNetCore,
    class_label: str,
    model: nn.Module = None,
    title: str = ""
    ) -> None:
    """Compares a clean point cloud (y) against its augmented/noisy version (x).

    Optionally, plots the comparison with a model's estimate f(x).
    """

    # sample random point cloud belonging to the class
    idx = np.random.choice(dataset.class_label_to_idx[class_label])
    transformation_type, noisy_point_cloud, target_point_cloud = dataset[idx][-3:]
    if dataset.input_transform:
        ratio_removed = dataset.input_transform.removal_amount
        noise_amount = dataset.input_transform.noise_amount
    else:
        ratio_removed = noise_amount = 0

    # # compute Chamfer Distance
    # if not title and torch.cuda.is_available:
    #     chamf_dist = chamfer_distance(
    #         p1=noisy_point_cloud.unsqueeze(0).cuda(),
    #         p2=target_point_cloud.unsqueeze(0).cuda()
    #     )
    #     title = f"Chamfer Distance: {round(chamf_dist.item(), 6)}"

    if transformation_type == 1:
        noisy_title = f"{int(ratio_removed * 100)}% points missing"
    elif transformation_type == 2:
        noisy_title = f"noise={noise_amount}"
    else:
        noisy_title = f"{int(ratio_removed * 100)}% points missing" + \
            ", " + f"noise amount = {noise_amount}"

    titles = ["Original", noisy_title]
    titles = titles + (["Reconstruction"] if model is not None else [])
    # make plotly subplots
    fig = make_subplots(
        rows=1, cols=2 if model is None else 3,
        specs=[[{'type': 'scatter3d'}] * (2 if model is None else 3)],
        subplot_titles=tuple(titles),
        horizontal_spacing=0.001
    )

    # create the scatter plots
    target_scatter_plot = go.Scatter3d(
        x=target_point_cloud[:, 0],
        y=target_point_cloud[:, 1],
        z=target_point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1.5)
    )
    noisy_scatter_plot = go.Scatter3d(
        x=noisy_point_cloud[:, 0],
        y=noisy_point_cloud[:, 1],
        z=noisy_point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1.5)
    )
    
    # run model inference
    if model is not None:
        device = list(model.parameters())[0].device
        reconstruction = model(
            noisy_point_cloud.unsqueeze(0).to(device)
        ).squeeze(0).cpu().detach()
        reconstruction_scatter_plot = go.Scatter3d(
            x=reconstruction[:, 0].numpy(),
            y=reconstruction[:, 1].numpy(),
            z=reconstruction[:, 2].numpy(),
            mode='markers',
            marker=dict(size=1.5)
        )
        print(torch.all(reconstruction[:512] == target_point_cloud[:512]), reconstruction.shape, target_point_cloud.shape)
        print(torch.all(noisy_point_cloud != 0, dim=-1).sum(), torch.all(reconstruction != 0, dim=-1).sum()) 
        print(torch.unique(reconstruction).shape, torch.unique(noisy_point_cloud).shape, torch.unique(target_point_cloud).shape)
        print(reconstruction.shape, noisy_point_cloud.shape, target_point_cloud.shape)
    else:
        reconstruction_scatter_plot = None


    # plot them
    fig.add_trace(target_scatter_plot, row=1, col=1)
    fig.add_trace(noisy_scatter_plot, row=1, col=2)
    if reconstruction_scatter_plot is not None:
        fig.add_trace(reconstruction_scatter_plot, row=1, col=3)

    # hides axes for cleaner look
    fig.update_layout(
        scene1=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        scene2=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        scene3=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title_text=title, title_x=0.5,
        autosize=True,
    )
    fig.show()
    

def visualize_point_cloud(point_cloud: Union[np.ndarray, torch.Tensor]) -> None:
    """Visualizes a point cloud."""
    # make plotly subplots
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]],
        subplot_titles=tuple(""),
        horizontal_spacing=0.001
    )

    # create the scatter plots
    scatter_plot = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1.5)
    )
    # plot them
    fig.add_trace(scatter_plot, row=1, col=1)

    # hides axes for cleaner look
    fig.update_layout(
        scene1=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title_text="", title_x=0.5,
        autosize=True,
    )
    fig.show()
    