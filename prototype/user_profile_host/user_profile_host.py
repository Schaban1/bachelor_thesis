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

from .recommender import *
from .optimizer import *
from .utils import constants
from diffusers import StableDiffusionPipeline




class UserProfileHost():
    def __init__(
            self, 
            original_prompt : str, 
            add_ons : list = None, 
            recommendation_type : str = 'bayes-opt', 
            optimization_type : str = 'gaussian_process', 
            hf_model_name : str ="stable-diffusion-v1-5/stable-diffusion-v1-5"
            ):
        
        # Some Clip Hyperparameters
        self.embedding_dim = 768
        self.n_clip_tokens = 77

        # Initialize tokenizer and text encoder to calculate CLIP embeddings
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=hf_model_name,
            cache_dir='./cache/'
        )
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.text_encoder.to(self.device)
        
        # Define the center of the user_space with the original prompt embedding
        self.center = self.clip_embedding(original_prompt)

        # Generate axis to define the user profile space with extensions of the original user-promt
        # by calculating the respective CLIP embeddings to the resulting prompts
        self.axis = []
        if not add_ons:
            add_ons = ['drawing', 'picture that old people like', 'funny', 'award-winning', 'highly detailed photoreal', 'aesthetic']
        for prompt in [original_prompt + ',' + add for add in add_ons]:
            self.axis.append(self.clip_embedding(prompt))
        self.num_axis = len(self.axis)

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = None

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Some Bayesian Optimization Hyperparameters
        self.num_steps = 5

        # Initialize an Optimizer
        if optimization_type == constants.MAX_PREF:
            self.optimizer = MaxPrefOptimizer()
        elif optimization_type == constants.WEIGHTED_SUM:
            self.optimizer = WeightedSumOptimizer()
        elif optimization_type == constants.GAUSSIAN_PROCESS:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"The optimization type {optimization_type} is not implemented yet.")

        # Initialize a Recommender
        if recommendation_type == constants.FUNCTION_BASED:
            self.recommender = BayesianRecommender()
        elif recommendation_type == constants.POINT:
            self.recommender = SinglePointRecommender()
        elif recommendation_type == constants.WEIGHTED_AXES:
            self.recommender = SinglePointWeightedAxesRecommender()
        else:
            raise ValueError(f"The recommendation type {recommendation_type} is not implemented yet.")



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

        self.user_profile = self.optimizer.optimize_user_profile(self.embeddings, self.preferences)

    def clip_embedding(self, prompt : str):
        '''
        Embeds a given prompt using CLIP.

        Returns:
            embedding (Tensor) : An embedding for the prompt in shape (1, 77, 768)
        '''
        prompt_tokens = self.tokenizer(prompt,
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",)

        prompt_embeds = self.text_encoder(prompt_tokens.input_ids.to(self.device))[0]
        return prompt_embeds.reshape(self.n_clip_tokens, self.embedding_dim)[0]

    def generate_recommendations(self, num_recommendations: int = 1, beta: float = None):
        '''
        This function generates recommendations based on the previously fit user-profile

        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Defines the trade-off between exploration and exploitation when using the BayesRecommender.
        Returns:
            embeddings (Tensor): Embeddings that can be retransformed into the CLIP space and used for image generation
        '''
        user_space_embeddings = self.recommender.recommend_embeddings(user_profile=self.user_profile, n_recommendations=num_recommendations)
        clip_embeddings = self.inv_transform(user_space_embeddings).reshape(num_recommendations, 1, self.embedding_dim).expand(num_recommendations, self.n_clip_tokens, self.embedding_dim)
        return clip_embeddings