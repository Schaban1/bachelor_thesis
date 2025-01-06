from abc import abstractmethod, ABC
import numpy as np
import torch
from torch import Tensor
from .utils import slerp

from botorch.acquisition import UpperConfidenceBound
from botorch.exceptions import InputDataWarning
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, LinearMean
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

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
        Hence, axes should be weighted differently according to their influence.
        There are two implementations: One where the points are on the surface of a sphere and one where they are not.
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


class RandomRecommender(Recommender):
    
    def __init__(self, n_embedding_axis, n_latent_axis, embedding_bounds=(0., 1.), latent_bounds=(-1., 1.)):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.embedding_bounds = embedding_bounds
        self.latent_bounds = latent_bounds

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """
        # Return random recommendations
        embed_alpha = torch.ones(self.n_embedding_axis)
        embed_distribution = torch.distributions.dirichlet.Dirichlet(embed_alpha)
        embeddings = embed_distribution.sample(sample_shape=(n_recommendations,))

        latent_alpha = torch.ones(self.n_latent_axis)
        latent_distribution = torch.distributions.dirichlet.Dirichlet(latent_alpha)
        latents = latent_distribution.sample(sample_shape=(n_recommendations,))
        factor = (torch.randint(low=0, high=2, size=(latents.shape[0],1)) * 2 - 1).expand(latents.shape[0], self.n_latent_axis)
        latents = latents * factor

        user_space_embeddings = torch.cat((embeddings, latents), dim=1)
        return user_space_embeddings


class SinglePointRecommender(Recommender):
    def __init__(self, embedding_bounds=(-1., 1.)):
        """
        :param embedding_bounds: Used to determine the radius used when embeddings lie on a sphere.
        """
        self.embedding_bounds = embedding_bounds

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
        radius = abs(self.embedding_bounds[0] - self.embedding_bounds[1]) / 2  # radius of sphere

        # recommendations on the surface of a sphere around the 0-center
        zero_centered_generated_points = self.get_random_samples_on_n_sphere(n_dims=len(user_profile), radius=radius,
                                                                             n_samples=n_recommendations)

        # move the points s.t. the user profile is the center
        return torch.add(zero_centered_generated_points, user_profile)


class SinglePointWeightedAxesRecommender(Recommender):

    def __init__(self, n_embedding_axis: int, n_latent_axis: int, embedding_bounds=(0., 1.), latent_bounds=(0., 1.),
                 exploration_factor: float = 1.0, on_sphere: bool = False):
        """
        :param n_embedding_axis: Number of axes in the embedding space.
        :param n_latent_axis: Number of axes in the latent space.
        :param latent_bounds: Bounds for the latent space.
        :param embedding_bounds: Bounds for the embedding space. The lower bound should not be smaller than 0, because
            experimental evaluation led to the findings that negative bounds produce noisy generated images.
        :param exploration_factor: Must be in [0, 1]. Determines factor multiplied to the chosen distance from the user
            profile, i.e. the magnitude of area searched. Values closer to 0 lead to smaller areas searched.
        :param on_sphere: Whether the recommendations should be on the surface of a sphere or not. By default, False.
            Experimental evaluation led to the findings that the generated images are too similar when the points are on
            the surface of a sphere.
        """
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.embedding_bounds = embedding_bounds
        self.latent_bounds = latent_bounds
        self.exploration_factor = exploration_factor    # TODO: decrease radius with higher iteration
        self.on_sphere = on_sphere
        # Define bounds for search space
        self.bounds = torch.tensor([
            # lower bounds (1, n_axis)
            [self.embedding_bounds[0] for i in range(self.n_embedding_axis)] + [self.latent_bounds[0] for i in
                                                                                range(self.n_latent_axis)],
            # upper bounds (1, n_axis)
            [self.embedding_bounds[1] for i in range(self.n_embedding_axis)] + [self.latent_bounds[1] for i in
                                                                                range(self.n_latent_axis)]
        ])

    def recommend_on_sphere(self, user_profile: Tensor, n_recommendations: int = 5, radius: float = 1.0) -> Tensor:
        """
        Uses SLERP to interpolate between the user profile and the axes of the user space.
        In this case, the points are on the surface of the sphere.
        The generated points are interpolated between the user profile and one axes.
        :param radius: Radius of the sphere.
        :param user_profile: Low-dimensional user profile.
        :param n_recommendations: Number of recommendations to return.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations on the surface of the sphere.
        """
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
        # whether recommendations should be on the sphere or not
        if self.on_sphere:  # usage of SLERP
            return self.recommend_on_sphere(user_profile, n_recommendations, radius=self.exploration_factor)

        axes = torch.eye(user_profile.shape[0])

        # distance of embedding bounds to user profile to find range to sample from
        lower_sampling_ranges = self.bounds[0] - user_profile
        upper_sampling_ranges = self.bounds[1] - user_profile

        # random weights for axes for each recommendation in bounds
        weights = torch.rand(size=(n_recommendations, user_profile.shape[0]))  # in [0, 1]
        # scale to bounds to ranges & scale with exploration factor
        weights = (self.exploration_factor *
                   (weights * (upper_sampling_ranges - lower_sampling_ranges) + lower_sampling_ranges))

        # interpolate between user profile and axes, user user_profile as reference point
        return user_profile + weights @ axes


class BayesianRecommender(Recommender):
    def __init__(self, n_embedding_axis, n_latent_axis, embedding_bounds=(0., 1.), latent_bounds=(-1., 1.), n_points_per_axis : int = 3, beta : float = 20, search_space_type : str = 'dirichlet'):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.embedding_bounds = embedding_bounds
        self.latent_bounds = latent_bounds
        self.cand_indices = []
        self.beta = beta
        self.reduce_beta = True
        self.n_points_per_axis = n_points_per_axis
        self.search_space_type = search_space_type

    def build_search_space(self):
        if self.search_space_type == 'dirichlet':
            n_samples = min(max((self.n_embedding_axis + self.n_latent_axis) * 5**((self.n_embedding_axis + self.n_latent_axis) // 2), 1000), 5000000)
            alpha = torch.ones(self.n_embedding_axis + self.n_latent_axis)
            dist = torch.distributions.dirichlet.Dirichlet(alpha)
            factor = torch.cat((torch.ones(n_samples, self.n_embedding_axis), (torch.randint(low=0, high=2, size=(n_samples,self.n_latent_axis)) * 2 - 1)), dim=1)
            search_space = dist.sample(sample_shape=(n_samples,)) * factor
            return search_space
        
        elif self.search_space_type == 'linspace':
            # Build search space and filter for embeddings that are on a sphere in CLIP space
            x_embed = torch.linspace(self.embedding_bounds[0], self.embedding_bounds[1], self.n_points_per_axis)
            x_latent = torch.linspace(self.latent_bounds[0], self.latent_bounds[1], self.n_points_per_axis)
            vectors = torch.meshgrid([x_embed for i in range(self.n_embedding_axis - 1)] + [x_latent for i in range(self.n_latent_axis)], indexing='ij')
            vectors = [v.flatten() for v in vectors]
            embed_grid = torch.stack(vectors[:self.n_embedding_axis-1], dim=1)
            latent_grid = torch.stack(vectors[self.n_embedding_axis-1:], dim=1)
            embed_grid_sum = torch.sum(embed_grid, dim=1)

            # Mask out the points where x + y > 1 to ensure they lie on the plane x + y + z = 1
            mask = embed_grid_sum <= 1

            # Get the corresponding z values
            final_embed_vector = 1 - embed_grid_sum

            # Apply the mask to filter out points outside the region 0 <= x + y + z <= 1
            embed_grid = embed_grid[mask]
            latent_grid = latent_grid[mask]
            final_embed_vector = final_embed_vector[mask]
            search_space = torch.cat((embed_grid, final_embed_vector.reshape(-1, 1), latent_grid), dim=-1)
            return search_space
        else:
            raise NotImplementedError('Invalid Search Space.')

    def recommend_embeddings(self, user_profile: Tensor = None, n_recommendations: int = 5) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        # Get embeddings and ratings from user profile
        embeddings, preferences = user_profile

        # Change Preference shape
        preferences = preferences.reshape(-1, 1)

        # Standardize embeddings
        mean, std = torch.mean(embeddings, dim=0), torch.std(embeddings, dim=0)
        embeddings_std = (embeddings - mean) / std

        # Build a search space for the BO to look in
        search_space = self.build_search_space()

        # Standardize search space
        search_space = (search_space - mean) / std

        # Get new acquisitions step by step
        for _ in range(n_recommendations):
            # Build a GP model
            mean_mod = ConstantMean()  # LinearMean(input_size=embeddings_std.shape[-1])
            model = SingleTaskGP(train_X=embeddings_std, train_Y=preferences, mean_module=mean_mod)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll = fit_gpytorch_mll(mll)

            # Initialize the acquisition function
            acqf = UpperConfidenceBound(model=model, beta=self.beta, maximize=True)

            # Get the highest scoring candidates out of meshgrid
            scores = acqf(search_space.reshape(search_space.shape[0], 1, search_space.shape[1]))
            candidate_idx = torch.argmax(scores)
            candidate = search_space[candidate_idx].reshape(1, -1)

            # Extend data with new candidate and predicted preference to include this information in the next iteration
            pseudo_preference = acqf._mean_and_sigma(X=candidate, compute_sigma=False)[0].detach()
            embeddings_std = torch.cat((embeddings_std, candidate))
            preferences = torch.cat((preferences, pseudo_preference.reshape(1, 1)))

        # Lower beta if settings require it
        if self.reduce_beta:
            self.beta -= 1

        # Return most promising candidates
        candidates_std = embeddings_std[-n_recommendations:]

        # Unstandardize and return them
        candidates = candidates_std * std + mean
        return candidates
