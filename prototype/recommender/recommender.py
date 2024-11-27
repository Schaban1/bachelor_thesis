from abc import abstractmethod, ABC
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prototype.utils.interpolation import slerp
import torch
from torch import Tensor
import prototype.utils.constants as constants
import prototype.utils.visualize_recommendations as visualize_recommendations


# WIP
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
        space and use this position to generate new generations, but one chooses multiple points in highly interesting
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

    def getRandomSamplesOnNSphere(self, n_dims: int = 10, radius: float = 1.0, n_samples: int = 5) -> Tensor:
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

        return torch.from_numpy(radius / np.sqrt(np.sum(x ** 2, 1, keepdims=True)) * x)


    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        """
        :param user_profile: A point in the low-dimensional user profile space.
        :param n_recommendations: Number of recommendations to return. By default, 5.
        :return: Tensor of shape (n_recommendations, n_dims) containing the samples on surface of sphere with center
            user_profile where n_dims is the dimensionality of the user_profile.
        """

        # recommendations on the surface of a sphere around the 0-center
        zero_centered_generated_points = self.getRandomSamplesOnNSphere(n_dims=len(user_profile),
                                                                        n_samples=n_recommendations)

        # move the points s.t. the user profile is the center
        return torch.add(zero_centered_generated_points, user_profile)




class SinglePointWeightedAxesRecommender(Recommender):

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        # hyperparameter
        radius = 1.0

        # weights for influence of axes: integer between 0 and n_recommendations
        weights = torch.randint(low=0, high=n_recommendations, size=(n_recommendations,))
            # torch.ones(user_profile.shape[0], dtype=int)   # equal weight
            #  # random weights for axes, TODO: where to get them from?

        # one hot encoding for axes
        axes = torch.multiply(torch.eye(user_profile.shape[0]), radius)
        print("axes", axes)
        print("weights", weights)

        # interpolate between user profile and axes
        # if fewer axes than n_recommendations, repeat axes
        interpolated_points = [slerp(user_profile, axis, num=n_recommendations)[weight] for axis, weight
                               in zip(axes.repeat(n_recommendations // axes.shape[0], 1), weights)]

        return torch.stack(interpolated_points)


class FunctionBasedRecommender(Recommender):

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        # TODO: Pauls BayesOpt approach
        pass




if __name__ == '__main__':
    dummy_user_profile = torch.tensor([1, 2, 3])  # torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # test single point recommender
    single_recommender = SinglePointRecommender()
    single_point_recommendations = single_recommender.recommend_embeddings(user_profile=dummy_user_profile,
                                                                           n_recommendations=200)
    if dummy_user_profile.shape[0] == 3:
        visualize_recommendations.display_generated_points(single_point_recommendations,
                                                           user_profile=dummy_user_profile)
    print("single point", single_point_recommendations)

    # test single point + weighted axes recommender
    single_weighted_recommender = SinglePointWeightedAxesRecommender()
    weighted_axes_recommendations = single_weighted_recommender.recommend_embeddings(user_profile=dummy_user_profile,
                                                                                     n_recommendations=100)
    if dummy_user_profile.shape[0] == 3:
        visualize_recommendations.display_generated_points(weighted_axes_recommendations,
                                                           user_profile=dummy_user_profile)
    print("single point + weighted axes", weighted_axes_recommendations)

    # test function-based recommender (Bayesian approach)
    # function_based_recommender = FunctionBasedRecommender()
    # function_based_recommendations = function_based_recommender.recommend_embeddings(user_profile=dummy_user_profile,
    #                                                                                  n_recommendations=100)
    # print("function_based_recommender", function_based_recommendations)

