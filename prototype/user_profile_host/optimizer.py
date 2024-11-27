from abc import abstractmethod, ABC
import torch
from torch import Tensor

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP


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
        user_profile = (preferences.reshape(-1) @ embeddings)/preferences.sum()
        return user_profile
    
class GaussianProcessOptimizer:

    def optimize_user_profile(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        """
        :param embeddings: The (user-space) embeddings of generated images the user saw and evaluated.
        :param preferences: The scores of the current user concerning the (user-space) embeddings.
        :return: A user profile that can be used by the recommender to generate new embeddings preferred by the user.
        """
        user_profile = SingleTaskGP(train_X=embeddings, train_Y=preferences)
        mll = ExactMarginalLogLikelihood(user_profile.likelihood, user_profile)
        mll = fit_gpytorch_mll(mll)
        return user_profile