from abc import abstractmethod, ABC
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prototype.utils.interpolation import slerp
import torch
from torch import Tensor
import prototype.utils.constants as constants
import matplotlib.pyplot as plt


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

    def display_generated_points(self, generated_points: Tensor):
        """
        Display the generated points in a 3D plot. The points are assumed to be in 3D.
        :param generated_points: Tensor of shape (n_points, 3) containing the generated points.
        :return: -
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for point in generated_points:
            ax.scatter(point[0], point[1], point[2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.title("Generated points on surface of sphere")
        plt.show()

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
        # TODO
        recommendations = []
        pass


class FunctionBasedRecommender(Recommender):

    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:
        # TODO: Pauls BayesOpt approach
        pass




if __name__ == '__main__':
    # test single point recommender
    single_recommender = SinglePointRecommender()
    dummy_user_profile = torch.tensor([1, 2, 3])  #torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    single_point_recommendations = single_recommender.recommend_embeddings(dummy_user_profile, n_recommendations=200)
    if dummy_user_profile.shape[0] == 3:
        single_recommender.display_generated_points(single_point_recommendations)
    # print("single point", single_point_recommendations)

    # single_weighted_recommender = SinglePointWeightedAxesRecommender()
    # print("single point + weighted axes",
    #       single_weighted_recommender.recommend_embeddings( [1, 2, 3, 4, 5]))

    # function_based_recommender = FunctionBasedRecommender()
    # print("function_based_recommender",
    #       function_based_recommender.recommend_embeddings([1, 2, 3, 4, 5]))

