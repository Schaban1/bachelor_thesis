'''
Based on the original prompt p_org, we can define our own coordinate system using the original prompt as the center (all zeros) and define
new axis by extending the prompt with additional attributes.

For example, given the original prompt p_org="A cute cat with a hat", we can define our own prompts like p_1="A cute cat with a hat, realistic" 
or p_2="A cute cat with a hat, old people like this image" and build a new coordinate system in the clip embeddings space defined by
emb(a_1, ..., a_n) = emb_org + a_1 * emb_1 + a_2 * emb_2 + ... + a_n * emb_n

where emb_i is the CLIP embedding of the respective prompt. With this, we redefine a new, n-dimensional space that can generate new CLIP embeddings
in the following way:
Let's say our user_profile is a vector in form of (a_1, ..., a_n), then we can sample around this point in the 10 dimensional space to get 
(b_1, ..., b_n) and inverse_transform it into the CLIP Embedding space, where it can either be used for image generation or creation of 
embeddings for image generation.

However, I feel like it would be smarter to consider a probability distribution over the user profile space as it allows us to draw samples at locations with
high variance (exploration) and high mean (exploitation). This would involve Bayesian Optimization and a multimodal gaussian distribution that will be fit by the optimizer 
and used for new recommendations by the recommender. Therefore, I think it would be easiest to store this inside one class, possibly the userProfileHost.

'''

import torch
from torch import Tensor

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from prototype.recommender import *
import prototype.utils.constants as constants


class UserProfileHost():
    def __init__(self, original_prompt : str, add_ons : list = None, recommendation_type : str ='bayes-opt'):
        self.center = self.clip_embedding(original_prompt)

        # Generate axis to define the user profile space with extensions of the original user-promt
        # by calculating the respective CLIP embeddings to the resulting prompts
        self.axis = []
        if not add_ons:
            add_ons = ['drawing', 'picture that old people like', 'funny', 'award-winning', 'highly detailed photoreal',
                       'aesthetic']
        for prompt in [original_prompt + ',' + add for add in add_ons]:
            self.axis.append(self.clip_embedding(prompt))
        self.num_axis = len(self.axis)

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = None

        # Placeholder until fit, then its a gaussian process regression
        self.user_profile = None

        # Defining how we will generate recommendations for the user
        self.recommendation_type = recommendation_type

        # Some Bayesian Optimization Hyperparameters
        self.num_steps = 5

    def inv_transform(self, user_embedding : Tensor):
        '''
        This function takes in a set of parameters [a_1, ..., a_n] and computes a respective CLIP embedding by using the dimensions provided in axis.

        Parameters:
            user_embedding (List[float]): Parameters concerning the initially defined axis of a user_embbing.
        '''
        clip_embedding = self.center
        for a_i, ax in zip(user_embedding, self.axis):
            clip_embedding += a_i * ax
        return clip_embedding

    def fit_user_profile(self, embeddings: Tensor, preferences: Tensor):
        '''
        This function initializes and fits a gaussian process for the available user preferences that can subsequently be used to 
        generate new interesting embeddings for the user.

        Parameters:
            embeddings (Tensor) : Embeddings that were presented to the user, where the embeddings are represented in the user-space.
            preferences (Tensor) : Preferences regarding the embeddings as real valued numbers.
        '''

        # Initialize or extend the available user related data
        if self.embeddings == None:
            self.embeddings = embeddings
            self.preferences = preferences
        else:
            self.embeddings = torch.cat((self.embeddings, embeddings))
            self.preferences = torch.cat((self.preferences, preferences))

        if self.recommendation_type == 'bayes-opt':
            # Initialize likelihood and model and train model on available data
            # TODO: Insert Normalization
            self.user_profile = SingleTaskGP(
                train_X=self.embeddings,
                train_Y=self.preferences
                )
            mll = ExactMarginalLogLikelihood(self.user_profile.likelihood, self.user_profile)
            mll = fit_gpytorch_mll(mll)
        else:
            # Use the highest valued user-space embedding as a user profile
            self.user_profile = self.embeddings[torch.argmax(self.preferences)]

            # TODO (Discuss): Weighted mean of all previously rated embeddings weighted by their value.
            # self.user_profile = (self.embeddings @ self.preferences)/self.preferences.sum()

    def clip_embedding(self, prompt):
        # TODO: Implement conversion from text to CLIP Embedding, should this be done here? Otherwise where can we get it?
        return torch.randn(size=(768,))  # Placeholder for tests

    def generate_recommendations(self, num_recommendations: int = 1, beta: float = 1):
        '''
        This function generates recommendations based on the previously fit user-profile

        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Defines the trade-off between exploration and exploitation.
        Returns:
            embeddings (List[tensor]): Embeddings that can be retransformed into the CLIP space and used for image generation
        '''
        if self.recommendation_type == constants.FUNCTION_BASED:
            if self.user_profile:  # Use the fittet gaussian process to evaluate which regions to sample next
                acqf = UpperConfidenceBound(self.user_profile, beta=beta)
                bounds = torch.stack([torch.zeros(self.num_axis), torch.ones(self.num_axis)])
                # TODO (Paul): Implement a method to ensure that the acquisition function only tests candidates that actually lead to useable image generations.
                xx = torch.linspace(start=0, end=1, steps=self.num_steps)
                mesh = torch.meshgrid([xx for i in range(self.num_axis)])
                mesh = torch.stack(mesh, dim=-1).reshape(self.num_steps**self.num_axis, 1, self.num_axis)
                scores = acqf(mesh)
                candidate_indices = torch.topk(scores, k=num_recommendations)[1]
                candidates = mesh[candidate_indices].reshape(num_recommendations, self.num_axis)
                return candidates
            else:  # If there is no user-profile available yet, return a number of random samples in the user-space
                return torch.rand(size=(num_recommendations, self.num_axis))
        else:
            if self.recommendation_type == constants.POINT:
                recommender = SinglePointRecommender()
            elif self.recommendation_type == constants.WEIGHTED_AXES:
                recommender = SinglePointWeightedAxesRecommender()
            else:
                raise ValueError(f"The recommendation type {self.recommendation_type} is not implemented yet.")

            return recommender.recommend_embeddings(user_profile=self.user_profile,
                                                    n_recommendations=num_recommendations)