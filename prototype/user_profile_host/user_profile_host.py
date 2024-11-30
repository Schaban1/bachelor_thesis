import torch
from torch import Tensor

from .recommender import *
from .optimizer import *
from ..constants import RecommendationType, OptimizationType
from diffusers import StableDiffusionPipeline


class UserProfileHost():
    def __init__(
            self, 
            original_prompt : str, 
            add_ons : list = None, 
            recommendation_type : str = RecommendationType.POINT, 
            optimization_type : str = OptimizationType.WEIGHTED_SUM, 
            hf_model_name : str ="stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir : str = './cache/'
            ):
        # Some Clip Hyperparameters
        self.embedding_dim = 768
        self.n_clip_tokens = 77

        # Initialize tokenizer and text encoder to calculate CLIP embeddings
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=hf_model_name,
            cache_dir=cache_dir
        )
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.text_encoder.to(self.device)
        pipe = None
        
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
        self.axis = torch.stack(self.axis)

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = None

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Some Bayesian Optimization Hyperparameters
        self.num_steps = 5

        # Initialize an Optimizer
        # TODO (Paul): Bayesian Optimization seems to sometimes fail when there are only few datapoints. Find a solution.
        if optimization_type == OptimizationType.MAX_PREF:
            self.optimizer = MaxPrefOptimizer()
        elif optimization_type == OptimizationType.WEIGHTED_SUM:
            self.optimizer = WeightedSumOptimizer()
        elif optimization_type == OptimizationType.GAUSSIAN_PROCESS:
            self.optimizer = GaussianProcessOptimizer()
        else:
            raise ValueError(f"The optimization type {optimization_type} is not implemented yet.")

        # Initialize a Recommender
        if recommendation_type == RecommendationType.FUNCTION_BASED:
            self.recommender = BayesianRecommender(n_steps=self.num_steps, n_axis=self.num_axis)
        elif recommendation_type == RecommendationType.POINT:
            self.recommender = SinglePointRecommender()
        elif recommendation_type == RecommendationType.WEIGHTED_AXES:
            self.recommender = SinglePointWeightedAxesRecommender()
        else:
            raise ValueError(f"The recommendation type {recommendation_type} is not implemented yet.")



    def inv_transform(self, user_embeddings : Tensor):
        '''
        This function transforms embeddings in the user_space back into the clip embedding space.

        Parameters:
            user_embedding (Tensor): Parameters concerning the initially defined axis of a user_embbing.

        Returns
            clip_embeddings (Tensor): The respective clip embeddings.
        '''
        clip_embeddings = self.center + (user_embeddings.to(self.device) @ self.axis)
        return clip_embeddings

    def fit_user_profile(self, preferences: Tensor):
        '''
        This function initializes and fits a gaussian process for the available user preferences that can subsequently be used to 
        generate new interesting embeddings for the user.

        Parameters:
            preferences (Tensor) : Preferences regarding the embeddings recommended last as real valued numbers.
        '''
        # Initialize or extend the available user related data
        if self.preferences == None:
            self.preferences = preferences
        else:
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
        This function generates recommendations based on the previously fit user-profile.

        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Defines the trade-off between exploration and exploitation when using the BayesRecommender.
        Returns:
            embeddings (Tensor): Embeddings that can be retransformed into the CLIP space and used for image generation
        '''
        if self.user_profile != None:
            user_space_embeddings = self.recommender.recommend_embeddings(user_profile=self.user_profile, n_recommendations=num_recommendations)
        else:
            # The zeros ensure that the original prompt embedding is included
            user_space_embeddings = torch.cat((torch.zeros(size=(1, self.num_axis)), torch.rand(size=(num_recommendations-1, self.num_axis))))
        
        if self.embeddings != None:
            self.embeddings = torch.cat((self.embeddings, user_space_embeddings)) # Safe the user_space_embeddings
        else:
            self.embeddings = user_space_embeddings
        clip_embeddings = self.inv_transform(user_space_embeddings).reshape(num_recommendations, 1, self.embedding_dim).cpu().expand(num_recommendations, self.n_clip_tokens, self.embedding_dim)
        return clip_embeddings