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
    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        pass


class NoOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        return (embeddings, preferences)
    
class SimpleOptimizer:
    def __init__(self, n_embedding_axis : int, n_latent_axis : int):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        beta = beta * 10

        # Create a probability distribution that handles the probabilites to select a certain embedding/latent
        embedding_idx, latent_idx = embeddings
        embedding_weights, latent_weights = [1 for _ in range(self.n_embedding_axis)], [1 for _ in range(self.n_latent_axis)]
        for i_emb, i_lat, p in zip(embedding_idx, latent_idx, preferences.reshape(-1).tolist()):
            embedding_weights[i_emb] += p * beta
            latent_weights[i_lat] += p * beta
        
        # Norm to get a probability distribution
        emb_sum = sum(embedding_weights)
        embedding_weights = [w/emb_sum for w in embedding_weights]
        lat_sum = sum(latent_weights)
        latent_weights = [w/lat_sum for w in latent_weights]
            
        return (embedding_weights, latent_weights)


class MaxPrefOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        user_profile = embeddings[torch.argmax(preferences)]
        return user_profile


class WeightedSumOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        if torch.count_nonzero(preferences) == 0:
            user_profile = None  # This will trigger the random recommender in embeddings
        else:
            user_profile = (preferences.reshape(-1) @ embeddings) / preferences.sum()
        return user_profile


class EMAWeightedSumOptimizer:

    def __init__(self, n_recommendations: int = 5, alpha: float = 0.2):
        """
        Initialize the EMAWeightedSumOptimizer. This optimizer uses an exponential moving average to update the user profile.
        :param n_recommendations: Number of recommendations to be considered recent each iteration.
        :param alpha: Factor for the exponential moving average. Higher values give more weight to recent recommendations.
        """
        self.n_recommendations = n_recommendations
        self.alpha = alpha

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        # If this is the first optimization step, just create a new user profile based on current embeddings
        if user_profile == None:
            if torch.count_nonzero(preferences) == 0:
                user_profile = None  # This will trigger the random recommender in embeddings
            else:
                user_profile = (preferences.reshape(-1) @ embeddings) / preferences.sum()
        else:
            new_embeddings, new_preferences = embeddings[-self.n_recommendations:], preferences[-self.n_recommendations:]
            # TODO (Discuss, Klara, Paul): whats the best response to all 0 preferences? Currently just keeping old user profile
            if not torch.count_nonzero(new_preferences) == 0:
                new_user_profile = (new_preferences.reshape(-1) @ new_embeddings) / new_preferences.sum()
                user_profile = self.alpha * new_user_profile + (1 - self.alpha) * user_profile

        return user_profile
