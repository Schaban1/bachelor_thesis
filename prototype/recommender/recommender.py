from abc import abstractmethod, ABC
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prototype.utils.interpolation import slerp
import torch
from torch import Tensor
import prototype.utils.constants as constants


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


    def recommend_embeddings(self, user_profile: Tensor, n_recommendations: int = 5) -> Tensor:

        # recommendations on the surface of a sphere around the user profile
        recommendations = []
        for i in range(n_recommendations):
            recommendations.append(slerp(user_profile, np.random.rand(1, len(user_profile)), 1)[0])


        return recommendations


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
    single_recommender = SinglePointRecommender()
    print("single point",
          single_recommender.recommend_embeddings([1, 2, 3, 4, 5]))
    single_weighted_recommender = SinglePointWeightedAxesRecommender()
    print("single point + weighted axes",
          single_weighted_recommender.recommend_embeddings( [1, 2, 3, 4, 5]))
    function_based_recommender = FunctionBasedRecommender()
    print("function_based_recommender",
          function_based_recommender.recommend_embeddings([1, 2, 3, 4, 5]))

