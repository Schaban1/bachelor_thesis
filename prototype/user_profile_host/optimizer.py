from abc import abstractmethod, ABC
import torch
from torch import Tensor


class Optimizer(ABC):  # ABC = Abstract Base Class
    """
    An optimizer class instance finds a matching user profile given embeddings and respective preferences from the current user.
    The resulting user_profile can then be used by the recommender to generate new embeddings that may fit the users taste/expectations.
    There are multiple types of optimization we consider for this:
    1. MAX_PREF optimizer returns the embedding in the user-space with the highest rating by the current user.
    2. WEIGHTED_SUM technically finds a single point in the user-space using a weighted sum of all available embeddings weighted by the users preferences.
        This should lead to the user-profile tending towards regions the user values higher than others.
    3. GAUSSIAN_PROCESS fits a gaussian process regressor using the available emeddings and user preferences. The resulting user_profile can then be used
        for Bayes optimization to evaluate a subspace of the user-space to look for places of high variance (uncertainty/new) and/or high mean (high valued
        regions of the user).
    """

    @abstractmethod
    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        pass


class NoOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        return (embeddings, preferences)


class MaxPrefOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        user_profile = embeddings[torch.argmax(preferences)]
        return user_profile


class WeightedSumOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        if torch.count_nonzero(preferences) == 0:  # if only zeros in preferences, black images are generated
            user_profile = torch.rand(size=(embeddings.shape[1],))  # fix: return a random user profile
        else:
            user_profile = (preferences.reshape(-1) @ embeddings) / preferences.sum()
        return user_profile


class EMAWeightedSumOptimizer:

    def __init__(self, n_recommendations: int = 5, alpha: int = 0.2):
        """
        Initialize the EMAWeightedSumOptimizer. This optimizer uses an exponential moving average to update the user profile.
        :param n_recommendations: Number of recommendations to be considered recent each iteration.
        :param alpha: Factor for the exponential moving average. Higher values give more weight to recent recommendations.
        """
        self.user_profile = None
        self.n_recommendations = n_recommendations
        self.alpha = alpha

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        # If this is the first optimization step, just create a new user profile based on current embeddings
        if self.user_profile == None:
            if torch.count_nonzero(preferences) == 0:
                self.user_profile = torch.rand(size=(embeddings.shape[1],))
            else:
                self.user_profile = (preferences.reshape(-1) @ embeddings) / preferences.sum()
        else:
            new_embeddings, new_preferences = embeddings[-self.n_recommendations:], preferences[
                                                                                    -self.n_recommendations:]
            # TODO (Discuss, Klara, Paul): whats the best response to all 0 preferences? Currently just keeping old user profile
            if not torch.count_nonzero(new_preferences) == 0:
                new_user_profile = (new_preferences.reshape(-1) @ new_embeddings) / new_preferences.sum()
                self.user_profile = self.alpha * new_user_profile + (1 - self.alpha) * self.user_profile

        return self.user_profile
