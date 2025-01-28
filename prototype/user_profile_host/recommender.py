from abc import abstractmethod, ABC
import numpy as np
import torch
from torch import Tensor

from botorch.acquisition import UpperConfidenceBound
from botorch.exceptions import InputDataWarning

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

import warnings

from .utils import get_unnormalized_value

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
    1. Single point generation with weighted axes:
        Some axes spanning the user space may convey more information than others.
        Hence, axes should be weighted differently according to their influence.
        There are two implementations: One where the points are on the surface of a sphere and one where they are not.
    2. Random generation:
        Random points in the user space are generated.
    3. Function-based generation:
        In this scenario, one doesn't want to optimize the position of the user profile (a point) in the suer profile
        space and use this position to generate new generations, but one chooses multiple points in fascinating
        regions of the user sspace and requests feedback of the user to "learn" the space.
        The choice of the points is based on an acquisition function, e.g. a Gaussian process.
    4. Dirichlet generation:
        A Dirichlet distribution is used to generate points in the user space.
        The concentration parameter is multiplied by the user profile to influence the distribution.
        The method increases the degree of exploitation with each recommendation.
        The Dirichlet distribution is chosen because it generates points on a simpley which produce roughly uniformly
        distributed samples in the high dimensional CLIP space after transformation & norming them.
    """

    @abstractmethod
    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        """
        :param user_profile: Encodes the user profile in the low-dimensional user profile space. Randomly initialized.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1].
        :return: A tensor of recommendations, i.e. n_recommendations many low-dimensional embeddings.
        """
        pass


class RandomRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = None) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :param beta: Not used in this recommender.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """
        # Return random recommendations
        alpha = torch.ones(self.n_axis)
        dist = torch.distributions.dirichlet.Dirichlet(alpha)
        random_user_embeddings = dist.sample(sample_shape=(n_recommendations,))
        return random_user_embeddings


class SinglePointWeightedAxesRecommender(Recommender):

    def __init__(self, n_embedding_axis: int, n_latent_axis: int):
        """
        :param n_embedding_axis: Number of axes in the embedding space.
        :param n_latent_axis: Number of axes in the latent space.
        :param latent_bounds: Bounds for the latent space.
        :param embedding_bounds: Bounds for the embedding space. The lower bound should not be smaller than 0, because
            experimental evaluation led to the findings that negative bounds produce noisy generated images.
        """
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.bounds = (0., 1.)

        # Define bounds for search space
        self.bounds = torch.tensor([
            # lower bounds (1, n_axis)
            [self.bounds[0] for i in range(self.n_embedding_axis)] + [self.bounds[0] for i in
                                                                                range(self.n_latent_axis)],
            # upper bounds (1, n_axis)
            [self.bounds[1] for i in range(self.n_embedding_axis)] + [self.bounds[1] for i in
                                                                                range(self.n_latent_axis)]
        ])

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = 0) -> Tensor:
        """
        Recommends embeddings based on the user profile, axes of the user space and the number of recommendations.
        Random weights are used to interpolate between the user profile and the axes.
        :param user_profile: Low-dimensional user profile.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        axes = torch.eye(user_profile.shape[0])

        # distance of embedding bounds to user profile to find range to sample from
        lower_sampling_ranges = self.bounds[0] - user_profile
        upper_sampling_ranges = self.bounds[1] - user_profile

        alpha = torch.ones(self.n_axis)  # Concentration parameter (uniform)
        distribution = torch.distributions.dirichlet.Dirichlet(alpha)
        weights_dirichlet = distribution.sample(sample_shape=(n_recommendations,))

        # scale to bounds to ranges & scale with exploration factor
        weights = ((1 - beta) * (
                weights_dirichlet * (upper_sampling_ranges - lower_sampling_ranges) + lower_sampling_ranges))

        # interpolate between user profile and axes, user user_profile as reference point
        return user_profile + weights @ axes


class DirichletRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis):
        """
        Initializes the Dirichlet Recommender.
        :param n_embedding_axis: Number of embedding axes (i.e. derived from prompt).
        :param n_latent_axis: Number of latent axes (i.e. 'noise').
        """
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5, beta: float = 0) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
        :return: Tensor of shape (n_recommendations, n_dims) containing the recommendations.
        """
        beta = get_unnormalized_value(beta, 1, 150)
        alpha = ((torch.ones(self.n_axis) * user_profile).reshape(-1) * beta)
        dist = torch.distributions.dirichlet.Dirichlet(alpha)
        search_space = dist.sample(sample_shape=(n_recommendations,))

        return search_space
    

class ClusterRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, beta: float = 1, beta_increase: float = 3):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.beta = beta
        self.beta_increase = beta_increase

    def recommend_embeddings(self, user_profile: Tensor = None, n_recommendations: int = 5, beta: float = None) -> Tensor:
        embeddings, preferences = user_profile
        top_embeddings = embeddings[preferences == preferences.max()]
        recommendations = []
        for i in range(n_recommendations):
            # Build Dirichlet Dist around this point
            alpha = (torch.ones(self.n_axis) * top_embeddings[i%top_embeddings.shape[0]]).reshape(-1) * (beta if beta else self.beta)
            dist = torch.distributions.dirichlet.Dirichlet(alpha)

            # Sample a point from Dist
            embed = dist.sample(sample_shape=(1,))

            # Add that to the recommendations
            recommendations.append(embed)

        recommendations = torch.cat(recommendations)

        # Increase beta
        beta += self.beta_increase
        return recommendations


class BayesianRecommender(Recommender):

    def __init__(self, n_embedding_axis, n_latent_axis, n_points_per_axis: int = 3):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_axis = n_embedding_axis + n_latent_axis
        self.n_points_per_axis = n_points_per_axis
        self.bounds = [0., 1.]

    def build_search_space(self):
        n_samples = min(max(self.n_axis * 5 ** (self.n_axis // 2), 1000), 500000)
        alpha = torch.ones(self.n_axis)
        dist = torch.distributions.dirichlet.Dirichlet(alpha)
        search_space = dist.sample(sample_shape=(n_samples,))
        return search_space


    def recommend_embeddings(self, user_profile: Tensor = None, n_recommendations: int = 5,
                             beta: float = 0.) -> Tensor:
        """
        Recommends embeddings based on the user profile, the number of recommendations and the trade-off between
        exploration and exploitation.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param n_recommendations: Number of recommendations to return.
        :param beta: Trade-off between exploration and exploitation. Higher beta means more exploitation.
            Must be in [0, 1]. 0 means full exploration, 1 means full exploitation.
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
            beta = 20 - get_unnormalized_value(beta, 0, 20)
            acqf = UpperConfidenceBound(model=model, beta=beta, maximize=True)

            # Get the highest scoring candidates out of meshgrid
            scores = acqf(search_space.reshape(search_space.shape[0], 1, search_space.shape[1]))
            candidate_idx = torch.argmax(scores)
            candidate = search_space[candidate_idx].reshape(1, -1)

            # Extend data with new candidate and predicted preference to include this information in the next iteration
            pseudo_preference = acqf._mean_and_sigma(X=candidate, compute_sigma=False)[0].detach()
            embeddings_std = torch.cat((embeddings_std, candidate))
            preferences = torch.cat((preferences, pseudo_preference.reshape(1, 1)))

        # Return most promising candidates
        candidates_std = embeddings_std[-n_recommendations:]

        # Unstandardize and return them
        candidates = candidates_std * std + mean
        return candidates

    def heat_map_values(self, user_profile: Tensor, user_space: Tensor, beta: float = 0.):
        """
        Get the values of the acquisition function for a given user profile and user space.
        These values can be used to create a heat map of the acquisition function, indicating which areas are likely to
        be chosen as future samples.
        :param user_profile: Low-dimensional user profile containing embeddings and preferences.
        :param user_space: The user space in which the user profile lies.
        :param beta: Trade-off between exploration and exploitation. Must be in [0, 1].
            0 means full exploration, 1 means full exploitation.
        :return: Tensor containing the values of the acquisition function.
        """
        # Get embeddings and ratings from user profile
        embeddings, preferences = user_profile

        # Change Preference shape
        preferences = preferences.reshape(-1, 1)

        # Standardize embeddings
        mean, std = torch.mean(embeddings, dim=0), torch.std(embeddings, dim=0)
        embeddings_std = (embeddings - mean) / std

        # Build a search space for the BO to look in
        search_space = user_space

        # Standardize search space
        search_space = (search_space - mean) / std

        # Build a GP model
        mean_mod = ConstantMean()  # LinearMean(input_size=embeddings_std.shape[-1])
        model = SingleTaskGP(train_X=embeddings_std, train_Y=preferences, mean_module=mean_mod)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = fit_gpytorch_mll(mll)

        # Initialize the acquisition function
        beta = 20 - get_unnormalized_value(beta, 0, 20)
        acqf = UpperConfidenceBound(model=model, beta=beta, maximize=True)

        # Get the highest scoring candidates out of meshgrid
        scores = acqf(search_space.reshape(search_space.shape[0], 1, search_space.shape[1])).detach()

        return scores
