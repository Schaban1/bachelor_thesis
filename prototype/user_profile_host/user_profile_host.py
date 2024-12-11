import torch
from torch import Tensor

from .recommender import *
from .optimizer import *
from ..constants import RecommendationType
from diffusers import StableDiffusionPipeline


class UserProfileHost():
    def __init__(
            self, 
            original_prompt : str, 
            add_ons : list = None,
            extend_original_prompt : bool = True,
            recommendation_type : str = RecommendationType.FUNCTION_BASED, 
            stable_dif_pipe : StableDiffusionPipeline = None,
            hf_model_name : str ="stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir : str = './cache/'
            ):
        # Some Clip Hyperparameters
        self.embedding_dim = 768
        self.n_clip_tokens = 77

        # Initialize tokenizer and text encoder to calculate CLIP embeddings
        if not stable_dif_pipe:     
            stable_dif_pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=hf_model_name,
                cache_dir=cache_dir
            )
        self.tokenizer = stable_dif_pipe.tokenizer
        self.text_encoder = stable_dif_pipe.text_encoder
        
        # Define the center of the user_space with the original prompt embedding
        self.center = self.clip_embedding(original_prompt)

        # Generate axis to define the user profile space with extensions of the original user-promt
        # by calculating the respective CLIP embeddings to the resulting prompts
        self.axis = []
        if not add_ons:
            add_ons = [
                'a detailed painting by hirohiko araki, featured on pixiv, analytical art, detailed painting, 2d game art, official art', 
                'realistic, colorful, 8k, highly detailed, trending on artstation', 
                'Extremely ultra-realistic photorealistic 3d, professional photography, natural lighting, volumetric lighting maximalist photo illustration 8k resolution detailed, elegant', 
                'by vincent van gogh',
                'flat, illustration, 4k',
                'captured in a painting with unparalleled detail and resolution at 64k',
                'Scratchy pen strokes, colored pen, blind contour, fisheye perspective close-up, stark hatch shaded sketchy scribbly, ink, strong angular shapes, woodcut shading, pen strokes, minimalist realistic, anime proportions, distorted perspective'
                'dramatic lighting, shot on leica, dark aesthetic',
                'detailed scene, red, perfect face, intricately detailed photorealism, trending on artstation, neon lights, rainy day, ray-traced environment, vintage 90s anime artwork',
                'in the style of pop art bold graphics, collage-based, cassius marcellus coolidge, aaron jasinski, peter blake, travel'
            ]
        if extend_original_prompt:
            for prompt in [original_prompt + ',' + add for add in add_ons]:
                self.axis.append(self.clip_embedding(prompt))
        else:
            for prompt in add_ons:
                self.axis.append(self.clip_embedding(prompt))
        self.num_axis = len(self.axis)
        self.axis = torch.stack(self.axis)

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = None

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Some Bayesian Optimization Hyperparameters
        self.bounds = (0. , 2.)

        # Initialize Optimizer and Recommender based on one Mode
        if recommendation_type == RecommendationType.FUNCTION_BASED:
            self.recommender = BayesianRecommender(n_axis=self.num_axis, bounds=self.bounds)
            self.optimizer = GaussianProcessOptimizer()
        elif recommendation_type == RecommendationType.POINT:
            self.recommender = SinglePointRecommender()
            self.optimizer = MaxPrefOptimizer()
        elif recommendation_type == RecommendationType.WEIGHTED_AXES:
            self.recommender = SinglePointWeightedAxesRecommender()
            self.optimizer = WeightedSumOptimizer()
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
        # r = n_rec
        # a = n_axis
        # t = n_tokens
        # e = embedding_size
        product = torch.einsum('ra,ate->rte', user_embeddings, self.axis)
        length = torch.linalg.vector_norm(self.center, ord=2, dim=-1, keepdim=False).reshape((1, product.shape[1], 1))
        total = (self.center + product)
        total = total / torch.linalg.vector_norm(total, ord=2, dim=-1, keepdim=True) * length
        return total

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

        prompt_embeds = self.text_encoder(prompt_tokens.input_ids)[0]
        return prompt_embeds.reshape(self.n_clip_tokens, self.embedding_dim)

    def generate_recommendations(self, num_recommendations: int = 1, beta: float = None):
        '''
        This function generates recommendations based on the previously fit user-profile.

        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Defines the trade-off between exploration and exploitation when using the BayesRecommender.
        Returns:
            embeddings (Tensor): Embeddings that can be retransformed into the CLIP space and used for image generation
        '''
        # Generate recommendations in the user_space
        if self.user_profile != None:
            user_space_embeddings = self.recommender.recommend_embeddings(user_profile=self.user_profile, n_recommendations=num_recommendations)
        else:
            # The zeros ensure that the original prompt embedding is included
            user_space_embeddings = torch.cat((torch.zeros(size=(1, self.num_axis)), torch.rand(size=(num_recommendations-1, self.num_axis))))*self.bounds[1]
        
        # Safe the user_space_embeddings
        if self.embeddings != None:
            self.embeddings = torch.cat((self.embeddings, user_space_embeddings)) 
        else:
            self.embeddings = user_space_embeddings

        # Transform embeddings from user_space to CLIP space
        clip_embeddings = self.inv_transform(user_space_embeddings).cpu()
        return clip_embeddings