import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interpolate
from torch import Tensor


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


def nlerp(v0, v1, t):
    l = lerp(v0, v1, t)
    return torch.nn.functional.normalize(l, dim=-1)


def slerp(v0, v1, t, DOT_THRESHOLD=0.99999):
    v0 = torch.nn.functional.normalize(v0, dim=-1)
    v1 = torch.nn.functional.normalize(v1, dim=-1)

    dot = (v0 * v1).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)  # Clamp for numerical stability

    theta_0 = torch.acos(dot)  # angle between v0 and v1
    sin_theta_0 = torch.sin(theta_0)

    # If the angle is small, use linear interpolation
    is_small_angle = dot.abs() > DOT_THRESHOLD

    slerp_result = (
            torch.sin((1 - t) * theta_0) / sin_theta_0 * v0 +
            torch.sin(t * theta_0) / sin_theta_0 * v1
    )

    return torch.where(is_small_angle, nlerp(v0, v1, t), slerp_result)



def display_generated_points(generated_points: Tensor, user_profile: Tensor, x_bounds=(-1, 1), y_bounds=(-1, 1),
                             z_bounds=(-1, 1)):
    """
    Display the generated points in a 3D plot. The points are assumed to be in 3D.
    :param generated_points: Tensor of shape (n_points, 3) containing the generated points.
    :param user_profile: The user profile in 3D.
    :param x_bounds: The bounds for the x-axis. Default is (-1, 1).
    :param y_bounds: The bounds for the y-axis. Default is (-1, 1).
    :param z_bounds: The bounds for the z-axis. Default is (-1, 1).
    :return: -
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim(z_bounds)

    for point in generated_points:
        ax.scatter(point[0], point[1], point[2])

    ax.scatter(user_profile[0], user_profile[1], user_profile[2], marker='x', color='red', s=50)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title("Generated points on surface of sphere")
    plt.show()


def display_generated_points_user_profile_2d(low_d_embeddings: Tensor, low_d_user_profile: Tensor, preferences: Tensor,
                                             compression_technique: str = "PCA", save_path: str = None):
    """
    Display the generated points in a 2D plot. The points are assumed to be in 2D.
    :param low_d_embeddings: Tensor of shape (n_points, 2) containing the generated points,
        i.e. points must be two-dimensional.
    :param low_d_user_profile: The user profile in 2D.
    :param preferences: (n_points,) preferences of the user concerning the generated points.
    :param compression_technique: The technique used to compress the points to 2D. Either "PCA" or "t-SNE".
    :param save_path: The path to save the plot to. Has to be a string and include "/" at the end. Default is None,
        hence, image is not saved.
    :return: -
    """
    cmap = plt.get_cmap('coolwarm')
    fig = plt.figure(figsize=(6, 5))
    p = plt.scatter(low_d_embeddings[:, 0], low_d_embeddings[:, 1], c=preferences, cmap=cmap)
    plt.colorbar(mappable=p)
    plt.scatter(low_d_user_profile[0], low_d_user_profile[1], c='black', s=150, marker='+', linewidth=3.5,
                label='User Profile')
    plt.legend()
    plt.title("Generated points up to now compressed to 2D using " + compression_technique)
    if save_path is not None:
        plt.savefig(save_path + f"generated_points_compressed_using_{compression_technique}.svg",
                    dpi=300, bbox_inches='tight', format='svg')
    plt.show()


def display_heatmap_user_profile_2d(low_d_embeddings: Tensor, grid_x: Tensor, grid_y: Tensor, scores: Tensor, preferences: Tensor,
                                             compression_technique: str = "PCA", save_path: str = None):
    """
    Display the generated points in a 2D plot. The points are assumed to be in 2D.
    :param low_d_embeddings: Tensor of shape (n_points, 2) containing the generated points,
        i.e. points must be two-dimensional.
    :param low_d_user_profile: The user profile in 2D.
    :param preferences: (n_points,) preferences of the user concerning the generated points.
    :param compression_technique: The technique used to compress the points to 2D. Either "PCA" or "t-SNE".
    :param save_path: The path to save the plot to. Has to be a string and include "/" at the end. Default is None,
        hence, image is not saved.
    :return: -
    """
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap('plasma')
    plt.scatter(low_d_embeddings[:, 0], low_d_embeddings[:, 1], edgecolors='black', c=preferences, cmap=cmap)
    p = plt.contourf(grid_x, grid_y, scores, alpha=.8, zorder=-1, cmap=cmap)
    plt.colorbar(mappable=p)
    plt.legend()
    plt.title("Generated points up to now compressed to 2D using " + compression_technique)
    if save_path is not None:
        plt.savefig(save_path + f"generated_points_compressed_using_{compression_technique}.svg",
                    dpi=300, bbox_inches='tight', format='svg')
    plt.show()

def get_unnormalized_value(x_norm:float, or_min:float, or_max:float):
    """
    Get the unnormalized value of x_norm given the original max and min values.
    :param x_norm: The normalized value.
    :param or_max: The original max value.
    :param or_min: The original min value.
    :return: The unnormalized value.
    """
    return x_norm * (or_max - or_min) + or_min
