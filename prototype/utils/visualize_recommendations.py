import matplotlib.pyplot as plt
from torch import Tensor

def display_generated_points(generated_points: Tensor, user_profile: Tensor):
    """
    Display the generated points in a 3D plot. The points are assumed to be in 3D.
    :param generated_points: Tensor of shape (n_points, 3) containing the generated points.
    :return: -
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for point in generated_points:
        ax.scatter(point[0], point[1], point[2])

    ax.scatter(user_profile[0], user_profile[1], user_profile[2], marker='x', color='red', s=50)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title("Generated points on surface of sphere")
    plt.show()