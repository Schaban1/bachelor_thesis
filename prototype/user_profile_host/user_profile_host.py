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
and used for new recommendations by the recommender. Therefore i think it would be easiest to store this inside one class, possibly the userProfileHost.

'''


import torch

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound


class UserProfileHost():
    def __init__(self, original_prompt, add_ons=None):
        self.center = self.clip_embedding(original_prompt)
        self.axis = [] # field to keep the original word embeddings that define the user-prompt. 
        if not add_ons:
            add_ons = ['realistic', 'drawing', 'picture that old people like', 'funny', 'award-winning']

        # Calculate the respective CLIP embeddings to the prompts that should define the user-space
        for prompt in [original_prompt +','+ add for add in add_ons]:
            self.axis.append(self.clip_embedding(prompt))

        self.num_axis = len(self.axis)

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = []
        self.preferences = []

        # Placeholder until fit, then its a gaussian process regression
        self.user_profile = None


    def inv_transform(self, user_embedding):
        '''
        This function takes in a set of parameters [a_1, ..., a_n] and computes a respective CLIP embedding by using the dimensions provided in axis.
        Parameters:
            user_embedding (List[float]): Parameters concerning the initially defined axis of a user_embbing.
        '''
        clip_embedding = self.center
        for a_i, ax in zip(user_embedding, self.axis):
            clip_embedding += a_i * ax
        return clip_embedding
    
    def fit_user_profile(self, embeddings, preferences):
        '''
        This function initializes and fits a gaussian process for the available user preferences that can subsequently be used to 
        generate new interesting embeddings for the user.
        '''
        # Extend the available user related data
        self.embeddings.extend(embeddings)
        self.preferences.extend(preferences)

        # Initialize likelihood and model and train model on available data
        self.user_profile = SingleTaskGP(self.embeddings, self.preferences)
        mll = ExactMarginalLogLikelihood(self.user_profile.likelihood, self.user_profile)
        mll = fit_gpytorch_mll(mll)

    def clip_embedding(self):
        # TODO: Implement conversion from text to CLIP Embedding, should this be done here?
        return None
    
    def generate_recommendations(self, num_recommendations: int = 1, beta: float = 1):
        '''
        This function generates recommendations based on the previously fit user-profile
        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Defines the trade-off between exploration and exploitation.
        Returns:
            embeddings (List[tensor]): Embeddings that can be retransformed into the CLIP space and used for image generation
        '''
        if self.user_profile: # Use the fittet gaussian process to evaluate which regions to sample next
            acqf = UpperConfidenceBound(self.user_profile, beta=beta)
            bounds = torch.stack([torch.zeros(self.num_axis), torch.ones(self.num_axis)])
            candidates, _ = optimize_acqf(
                acqf, bounds=bounds, q=num_recommendations, num_restarts=5, raw_samples=50,
            )
            return candidates
        else: # If there is no user-profile available yet, return a number of random samples
            return torch.rand(size=(num_recommendations,self.num_axis))


if __name__ == '__main__':
    # TODO: Write some tests.
    print('test')
