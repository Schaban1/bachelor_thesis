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

    def __init__(self, n_embedding_axis : int, n_latent_axis : int, image_styles : list, secondary_contexts : list, atmospheric_attributes : list, quality_terms : list):
        self.n_embedding_axis = n_embedding_axis
        self.n_latent_axis = n_latent_axis
        self.n_image_styles = len(image_styles)
        self.n_secondary_contexts = len(secondary_contexts)
        self.n_atmospheric_attributes = len(atmospheric_attributes)
        self.n_quality_terms = len(quality_terms)
        self.beta_factor = 10

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor, user_profile: Tensor, beta : float = None) -> Tensor:
        """
        :param embeddings: A list of lists containing the indices chosen for each prompt part
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        # TODO: Find optimal beta_factor value
        beta = beta * self.beta_factor

        # Create a probability distribution that handles the probabilites to select a certain term/latent
        img_idx, sec_idx, at_idx, qual_idx, lat_idx = embeddings
        img_votes = [1 for _ in range(self.n_image_styles)]
        sec_votes = [1 for _ in range(self.n_secondary_contexts)]
        at_votes = [1 for _ in range(self.n_atmospheric_attributes)]
        qual_votes = [1 for _ in range(self.n_quality_terms)]
        lat_votes = [1 for _ in range(self.n_latent_axis)]

        print('Debug prints in Optimizer')
        print('img_idx', img_idx)

        # Add preference votes on the individual terms/latents
        for i_img, i_sec, i_at, i_qual, i_lat, p in zip(img_idx, sec_idx, at_idx, qual_idx, lat_idx, preferences.reshape(-1).tolist()):
            img_votes[i_img] += p * beta
            sec_votes[i_sec] += p * beta
            at_votes[i_at] += p * beta
            qual_votes[i_qual] += p * beta
            lat_votes[i_lat] += p * beta

        print("Image Votes (for debugging): ",img_votes)
        
        # Norm to get a probability distribution
        img_sum = sum(img_votes)
        img_weights = [v/img_sum for v in img_votes]
        sec_sum = sum(sec_votes)
        sec_weights = [v/sec_sum for v in sec_votes]
        at_sum = sum(at_votes)
        at_weights = [v/at_sum for v in at_votes]
        qual_sum = sum(qual_votes)
        qual_weights = [v/qual_sum for v in qual_votes]
        lat_sum = sum(lat_votes)
        lat_weights = [v/lat_sum for v in lat_votes]
        
        # Return weights for drawing new terms to be included in 
        return (img_weights, sec_weights, at_weights, qual_weights, lat_weights)


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
