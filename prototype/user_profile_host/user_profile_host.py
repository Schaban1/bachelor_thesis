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
            cache_dir : str = './cache/',
            n_embedding_axis : int = 10,
            embedding_bounds : tuple = (0., 1.),
            use_embedding_center: bool = True,
            n_latent_axis : int = 2,
            latent_bounds : tuple = (1., 5.),
            use_latent_center: bool = True,
            ):
        # Some Clip Hyperparameters
        self.embedding_dim = 768
        self.n_clip_tokens = 77
        self.height = 512
        self.width = 512
        self.latent_space_length = 15.55
        self.n_latent_axis = n_latent_axis
        self.n_embedding_axis = n_embedding_axis
        self.use_embedding_center = use_embedding_center
        self.user_latent_center = use_latent_center

        # Initialize tokenizer and text encoder to calculate CLIP embeddings
        if not stable_dif_pipe:     
            stable_dif_pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=hf_model_name,
                cache_dir=cache_dir
            )
        self.tokenizer = stable_dif_pipe.tokenizer
        self.text_encoder = stable_dif_pipe.text_encoder
        
        # Define the center of the user_space with the original prompt embedding
        self.embedding_center = self.clip_embedding(original_prompt)
        self.embedding_length = torch.linalg.vector_norm(self.embedding_center, ord=2, dim=-1, keepdim=False)
        if not use_latent_center:
            self.embedding_center = torch.zeros(size=(1, self.n_clip_tokens, self.embedding_dim))

        # Generate axis to define the user profile space with extensions of the original user-promt
        # by calculating the respective CLIP embeddings to the resulting prompts
        self.embedding_axis = []
        if not add_ons:
            add_ons = [
                'a detailed painting by hirohiko araki, featured on pixiv, analytical art, detailed painting, 2d game art, official art', 
                'realistic, colorful, 8k, highly detailed, trending on artstation', 
                'Extremely ultra-realistic photorealistic 3d, professional photography, natural lighting, volumetric lighting maximalist photo illustration 8k resolution detailed, elegant', 
                'captured in a painting with unparalleled detail and resolution at 64k',
                'Scratchy pen strokes, colored pen, blind contour, fisheye perspective close-up, stark hatch shaded sketchy scribbly, ink, strong angular shapes, woodcut shading, pen strokes, minimalist realistic, anime proportions, distorted perspective',
                'dramatic lighting, shot on leica, dark aesthetic',
                'detailed scene, red, intricately detailed photorealism, trending on artstation, neon lights, rainy day, ray-traced environment, vintage 90s anime artwork',
                'in the style of pop art bold graphics, collage-based, cassius marcellus coolidge, aaron jasinski, peter blake, travel',
                'highly textured, genre-defining mixed media collage painting, fringe absurdism, award-winning halftone pattern illustration, simple flowing shapes, subtle shadows, paper texture, minimalist color scheme, inspired by zdzisław beksiński',
                "full body, unreal, created by alberto seveso, ethereal, featuring an optical illusion, mystical, luminous, with twinkling lights, surreal, showcasing 3d fractals, high resolution, sharp details, soft, with a dreamy glow, translucent, water drops, in 8k resolution, resembling a nebula, beautiful, with a broken glass effect, without a background, stunning, representing something that doesn't even exist, a mythical being exuding energy, textures, iridescent and luminescent scales, breathtaking beauty, pure perfection, with a divine presence, unforgettable, impressive"
            ][:self.n_embedding_axis]
        if extend_original_prompt:
            for prompt in [original_prompt + ',' + add for add in add_ons]:
                self.embedding_axis.append(self.clip_embedding(prompt))
        else:
            for prompt in add_ons:
                self.embedding_axis.append(self.clip_embedding(prompt))

        self.embedding_axis = torch.stack(self.embedding_axis)
        if n_latent_axis:
            self.latent_center = torch.randn((1, stable_dif_pipe.unet.config.in_channels, self.height // 8, self.width // 8)) if use_latent_center else torch.zeros(size=(1, stable_dif_pipe.unet.config.in_channels, self.height // 8, self.width // 8))
            self.latent_axis = torch.randn((n_latent_axis, stable_dif_pipe.unet.config.in_channels, self.height // 8, self.width // 8))
            self.num_axis = self.embedding_axis.shape[0] + self.latent_axis.shape[0]
        else:
            self.num_axis = self.embedding_axis.shape[0]

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = None

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Some Bayesian Optimization Hyperparameters
        self.embedding_bounds = embedding_bounds
        self.latent_bounds = latent_bounds

        # Initialize Optimizer and Recommender based on one Mode
        if recommendation_type == RecommendationType.FUNCTION_BASED:
            self.recommender = BayesianRecommender(n_embedding_axis=self.n_embedding_axis, n_latent_axis=self.n_latent_axis, embedding_bounds=self.embedding_bounds, latent_bounds=latent_bounds)
            self.optimizer = NoOptimizer()
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
        if self.n_latent_axis:
            latent_factors = user_embeddings[:,-self.latent_axis.shape[0]:]
            user_embeddings = user_embeddings[:,:-self.latent_axis.shape[0]]

        # r = n_rec, a = n_axis, t = n_tokens, e = embedding_size
        product = torch.einsum('ra,ate->rte', user_embeddings, self.embedding_axis)
        embedding_length = self.embedding_length.reshape((1, product.shape[1], 1))
        clip_embeddings = (self.embedding_center + product)
        clip_embeddings = clip_embeddings / torch.linalg.vector_norm(clip_embeddings, ord=2, dim=-1, keepdim=True) * embedding_length

        latents = None
        if self.n_latent_axis:
            latents = self.latent_center + torch.einsum('rl,lxyz->rxyz', latent_factors, self.latent_axis)
            latents = latents / torch.linalg.matrix_norm(latents, ord=2, dim=(-2, -1), keepdim=True) * self.latent_space_length

        return clip_embeddings, latents
    
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
                            return_tensors="pt",).to(self.text_encoder.device)

        prompt_embeds = self.text_encoder(prompt_tokens.input_ids)[0].cpu()
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
            # Start initially with some random embeddings
            user_space_embeddings = torch.rand(size=(num_recommendations, self.num_axis))
        
        # Safe the user_space_embeddings
        if self.embeddings != None:
            self.embeddings = torch.cat((self.embeddings, user_space_embeddings)) 
        else:
            self.embeddings = user_space_embeddings

        # Transform embeddings from user_space to CLIP space
        clip_embeddings, latents = self.inv_transform(user_space_embeddings)
        return clip_embeddings, latents