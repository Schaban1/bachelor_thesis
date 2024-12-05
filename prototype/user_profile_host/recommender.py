from abc import abstractmethod, ABC
import numpy as np
import torch
from torch import Tensor
from .utils import slerp
from botorch.acquisition import UpperConfidenceBound
from botorch.exceptions import InputDataWarning
import warnings

warnings.simplefilter("ignore", category=InputDataWarning)


class Recommender(ABC):  # ABC = Abstract Base Class
    """
    A Recommender class instance derives recommended samples for the next iteration.
    In other words:
    Multiple alterations of the current user profile (first iteration: random) are returned.
    It is important to note, that the user profile is a vector in the user space,
    i.e. a low dimensional subspace of the CLIP space.
    Hence, one assumes that points generated in the user space which are projected back to the CLIP space
    correspond to valid embeddings, i.e. produce meaningful images.
    The method used for generation depends on the user choice.
    It is possible to generate embeddings via the following methods:
    1. Single point generation:
        Multiple random points on a sphere surface around the current user profile are generated.
        These points are returned as recommendation.
    2. Single point generation with weighted axes:
        Some axes spanning the user space may convey more information than others.
        Hence, highly influential axes should be weighted less than others to counteract this phenomenon.
    3. Function-based generation:
        In this scenario, one doesn't want to optimize the position of the user profile (a point) in the suer profile
        space and use this position to generate new generations, but one chooses multiple points in fascinating
        regions of the user sspace and requests feedback of the user to "learn" the space.
        The choice of the points is based on an acquisition function, e.g. a Gaussian process.
    """

    @abstractmethod
    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        """
        :param user_profile: Encodes the user profile in the low-dimensional user profile space. Randomly initialized.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :return: A tensor of recommendations, i.e. n_recommendations many low-dimensional embeddings.
        """
        pass


class SinglePointRecommender(Recommender):

    def get_random_samples_on_n_sphere(self, n_dims: int = 10, radius: float = 1.0, n_samples: int = 5) -> Tensor:
        """
        Code from: https://stackoverflow.com/questions/52808880/algorithm-for-generating-uniformly-distributed-random
        -points-on-the-n-sphere (27.11.2024)
        Idea from: https://mathworld.wolfram.com/HyperspherePointPicking.html (27.11.2024)

        :param n_dims: Number of dimensions of system. The sphere surface is n_dims-1 dimensional.
        :param radius: Radius of the sphere.
        :param n_samples: Number of samples to generate.
        :return: Tensor of shape (n_samples, n_dims) containing the samples on surface of sphere with center 0^n_dims.
        """
        x = np.random.default_rng().normal(size=(n_samples, n_dims))

        return torch.from_numpy(radius / np.sqrt(np.sum(x ** 2, 1, keepdims=True)) * x).float()

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """

        # recommendations on the surface of a sphere around the 0-center
        zero_centered_generated_points = self.get_random_samples_on_n_sphere(n_dims=len(user_profile),
                                                                             n_samples=n_recommendations)

        # move the points s.t. the user profile is the center
        return torch.add(zero_centered_generated_points, user_profile)


class SinglePointWeightedAxesRecommender(Recommender):

    def recommend_on_sphere(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        """
        Uses SLERP to interpolate between the user profile and the axes of the user space.
        In this case, the points are on the surface of the sphere.
        The generated points are interpolated between the user profile and one axes.
        :param user_profile: Low-dimensional user profile.
        :param n_recommendations: Number of recommendations to return.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations on the surface of the sphere.
        """
        # hyperparameter
        radius = 1.0

        # weights for influence of axes: random integer between 0 and n_recommendations
        weights = torch.randint(low=0, high=n_recommendations, size=(n_recommendations,))  # random weights for axes

        # one hot encoding for axes
        axes = torch.multiply(torch.eye(user_profile.shape[0]), radius)

        # interpolate between user profile and axes:
        # Weights are used to determine the influence of the axes (SLERP returns interpolated points with increasing
        # influence of the axes) -> the higher the weight (i.e. index), the more the axis is taken into account
        # if fewer axes than n_recommendations, repeat axes
        interpolated_points = [slerp(user_profile, axis, num=n_recommendations)[weight] for axis, weight
                               in zip(axes.repeat(n_recommendations // axes.shape[0], 1), weights)]

        return torch.stack(interpolated_points)

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        """
        Recommends embeddings based on the user profile, axes of the user space and the number of recommendations.
        If points should be on the sphere, SLERP is used to interpolate between the user profile and the axes.
        Otherwise, random weights are used to interpolate between the user profile and the axes.
        :param user_profile: Low-dimensional user profile.
        :param n_recommendations: Number of recommendations to return.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        on_sphere = False  # whether recommendations should be on the sphere or not
        axes = torch.eye(user_profile.shape[0])

        if on_sphere:  # usage of SLERP
            return self.recommend_on_sphere(user_profile, n_recommendations)

        matrix = torch.cat((axes, user_profile.unsqueeze(0)), dim=0)

        # random weights for axes for each recommendation
        weights = torch.rand(size=(n_recommendations, user_profile.shape[0] + 1))
        weights /= torch.sum(weights, dim=1, keepdim=True)  # normalize weights

        interpolated_points = [torch.from_numpy(
            np.einsum('i,ij->ij', weight, matrix)).sum(axis=0)
                               for weight in weights]

        return torch.stack(interpolated_points)


class BayesianRecommender(Recommender):
    def __init__(self, n_steps, n_axis, bounds=(0,1)):
        self.n_steps = n_steps
        self.n_axis = n_axis
        self.bounds = bounds
        self.cand_indices = []

    def recommend_embeddings(self, user_profile: Tensor = None, n_recommendations: int = 5, beta: float = 1) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        acqf = UpperConfidenceBound(user_profile, beta=beta)
        xx = torch.linspace(start=self.bounds[0], end=self.bounds[1], steps=self.n_steps)
        mesh = torch.meshgrid([xx for i in range(self.n_axis)], indexing="ij")
        mesh = torch.stack(mesh, dim=-1).reshape(self.n_steps**self.n_axis, 1, self.n_axis)


        # Get highest scoring candidates out of meshgrid
        scores = acqf(mesh)
        candidate_indices = torch.topk(scores, k=n_recommendations+len(self.cand_indices))[1]

        # Remove indices that have already been sampled
        candidate_indices = [i for i in candidate_indices if i not in self.cand_indices][:n_recommendations]

        # Return most promising candidates
        candidates = mesh[candidate_indices].reshape(n_recommendations, self.n_axis)
        return candidates                
