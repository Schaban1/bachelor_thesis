import torch
from torch import Tensor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nicegui import binding

from .recommender import *
from .optimizer import *
from ..constants import RecommendationType
from diffusers import StableDiffusionPipeline


class UserProfileHost():
    original_prompt = binding.BindableProperty()
    extend_original_prompt = binding.BindableProperty()
    recommendation_type = binding.BindableProperty()
    height = binding.BindableProperty()
    width = binding.BindableProperty()
    latent_space_length = binding.BindableProperty()
    n_latent_axis = binding.BindableProperty()
    n_embedding_axis = binding.BindableProperty()
    use_embedding_center = binding.BindableProperty()
    use_latent_center = binding.BindableProperty()
    n_recommendations = binding.BindableProperty()
    ema_alpha = binding.BindableProperty()
    beta = binding.BindableProperty()
    beta_step_size = binding.BindableProperty()

    # TODO: Group together Recommender Args and just pass them to the recommender, should simplyfy this arg list
    def __init__(
            self,
            original_prompt: str,
            add_ons: list = None,
            extend_original_prompt: bool = True,
            recommendation_type: str = RecommendationType.RANDOM,
            stable_dif_pipe: StableDiffusionPipeline = None,
            hf_model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir: str = './cache/',
            n_embedding_axis: int = 13,
            use_embedding_center: bool = True,
            n_latent_axis: int = 3,
            use_latent_center: bool = False,
            n_recommendations: int = 5,
            ema_alpha: float = 0.5,
            beta: float = 0.,
            beta_step_size: float = 0.1,
    ):
        """
        This class is the main interface for the user profile host. It initializes the user profile host with the
        :param original_prompt: The original prompt as string.
        :param add_ons: A list of additional prompts to be used as axis for the user profile space.
            Elements of the list are strings.
        :param extend_original_prompt: Whether to extend the original prompt with the add_ons, separated by ', '.
        :param recommendation_type: The type of recommender to be used. Must be in constants.RecommendationType
        :param stable_dif_pipe: If given, the pipeline will be used to calculate the CLIP embeddings.
        Otherwise, a new pipeline will be created.
        :param hf_model_name: Name of the Hugging Face model.
        :param cache_dir: Path to the cache directory.
        :param n_embedding_axis: Number of axis to be used for the user profile.
        :param use_embedding_center: Whether to use the original prompt as the center of the user profile space.
        :param n_latent_axis: Number of latent axis to be used for the user profile.
        :param use_latent_center: Whether to use a latent center instead of all zeros.
        :param n_recommendations: Number of recommendations to be generated each iteration.
        :param ema_alpha: Used for an exponential moving average to update the user profile.
            Factor for the exponential moving average. Higher values give more weight to recent recommendations.
        :param weighted_axis_beta: Used for the weighted axes recommender. 0 -> high exploration, 1 -> high exploitation
        :param bo_beta: initial beta for BayesianRecommender
        :param di_beta: initial beta for DirichletRecommender
        :param di_beta_increase: increase beta by this amount after each iteration (DirichletRecommender)
        """
        # Some Clip Hyperparameters
        self.original_prompt = original_prompt
        self.add_ons = add_ons
        self.extend_original_prompt = extend_original_prompt
        self.recommendation_type = recommendation_type
        self.stable_dif_pipe = stable_dif_pipe
        self.embedding_dim = 768
        self.n_clip_tokens = 77
        self.height = 512
        self.width = 512
        self.latent_space_length = 15.55
        self.n_latent_axis = n_latent_axis
        self.n_embedding_axis = n_embedding_axis
        self.use_embedding_center = use_embedding_center
        self.use_latent_center = use_latent_center
        self.n_recommendations = n_recommendations
        self.ema_alpha = ema_alpha
        self.beta = min(beta, 1.)
        self.beta_step_size = beta_step_size

        # Check for valid values
        assert self.beta >= 0., "Beta should be in range [0., 1.]"
        assert self.beta_step_size >= 0. and self.beta_step_size < 1., "Beta Step Size should be in [0., 1.]"

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = None

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Bounds remain fixed to 0., 1. for simplicity
        self.embedding_bounds = [0., 1.]
        self.latent_bounds = [0., 1.]

        # Initialize tokenizer and text encoder to calculate CLIP embeddings
        if not self.stable_dif_pipe:
            self.stable_dif_pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=hf_model_name,
                cache_dir=cache_dir
            )
        self.tokenizer = self.stable_dif_pipe.tokenizer
        self.text_encoder = self.stable_dif_pipe.text_encoder

        self.load_user_profile_host()

    def load_user_profile_host(self):
        # Define the center of the user_space with the original prompt embedding
        self.embedding_center = self.clip_embedding(self.original_prompt)
        self.embedding_length = torch.linalg.vector_norm(self.embedding_center, ord=2, dim=-1, keepdim=False)
        if not self.use_embedding_center:
            self.embedding_center = torch.zeros(size=(1, self.n_clip_tokens, self.embedding_dim))

        # Generate axis to define the user profile space with extensions of the original user-promt
        # by calculating the respective CLIP embeddings to the resulting prompts
        # TODO: Discuss, if this could be improved.
        self.embedding_axis = []
        if not self.add_ons:
            self.add_ons = [
                               "beautiful, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a cold snowstorm, fujifilm xt3, outdoors, beautiful lighting, raw photo, 8k uhd, film grain, unreal engine 5, ray trace",
                               "in the style of liquid metal, vray tracing, raw character, 32k uhd, schlieren photography, conceptual portraiture, wet - on - wet blending",
                               "Detailed, vibrant illustration, full of plants, trees, by herge, in the style of tin-tin comics, vibrant colors, detailed, lots of people, sunny day, beautiful illustration",
                               "Sun profile, halftone pattern, editorial illustration of the memento morti, higly textured, genre defining mixed media collage painting, fringe absurdism, award winning halftone pattern illustration, simple flowing shapes, subtle shadows, paper texture, minimalist color scheme, inspired by zdzisław beksiński",
                               "3d illustration, in the style of fantasy, minimalistic, featuring multiple soft and rounded fractal, complex forms dressed as royal and glamorous in gold and white",
                               "Black, thin lines, all lines have the same mass and weight, continuity can be seen in the common flow of all lines, the lines occupy only the central part of the image, white background",
                               "a detailed painting by hirohiko araki, featured on pixiv, analytical art, detailed painting, 2d game art, official art",
                               "A surreal picture, in the style of pop art bold graphics, collage-based, cassius marcellus coolidge, aaron jasinski, peter blake, travel, nyc explosion coverage",
                               "Realistic, red white and black, made of red coral, mahogany, black obsidian, bloodstone, tourmaline and gold, elegant, diamonds, gold, elegant, masterpiece, concept art, tectonic, gold shiny background, nikon photography, shot photography by wes anderson, kodak color, hd, 300mm",
                               "colored ink mikhail garmash, louis jover, victor cheleg, damien hirst, ivan aizovsky, claude joseph vernet, broken glass effect, no background, amazing, something that doesn’t even exist, mythical creature, energy, molecular, textures, shimmering and luminescent colors, breathtaking beauty, pure perfection, divine presence, unforgettable, impressive, three-dimensional light, auras, rays, vibrant colors, broken glass effect, no background, stunning, something that even doesn't exist, mythical being, energy, molecular, textures, iridescent and luminescent scales, breathtaking beauty, pure perfection, divine presence, unforgettable, impressive, breathtaking beauty, volumetric light, auras, rays, vivid colors reflects",
                               "shot on leica, shadowplay, gorgeous lighting, subtle pastel hues, 8k, pretty freckles",
                               "behind windwow, rainy, black and white photography surreal art blurry minimalistic",
                               "at full height, is working on a beautiful design project, creating design projects, a beautiful workspace, aesthetics, correct proportions realism ultra high quality, real photo"
                           ][:self.n_embedding_axis]
        if self.extend_original_prompt:
            for prompt in [self.original_prompt + ', ' + add for add in self.add_ons]:
                self.embedding_axis.append(self.clip_embedding(prompt))
        else:
            for prompt in self.add_ons:
                self.embedding_axis.append(self.clip_embedding(prompt))

        self.embedding_axis = torch.stack(self.embedding_axis)
        if self.n_latent_axis:
            self.latent_center = torch.randn((1, self.stable_dif_pipe.unet.config.in_channels, self.height // 8,
                                              self.width // 8)) if self.use_latent_center else (
                torch.zeros(size=(1, self.stable_dif_pipe.unet.config.in_channels, self.height // 8, self.width // 8)))
            self.latent_axis = torch.randn(
                (self.n_latent_axis, self.stable_dif_pipe.unet.config.in_channels, self.height // 8,
                 self.width // 8))
            self.num_axis = self.embedding_axis.shape[0] + self.latent_axis.shape[0]
        else:
            self.num_axis = self.embedding_axis.shape[0]

        # Generally required througout this programm
        self.random_recommender = RandomRecommender(n_embedding_axis=self.n_embedding_axis, n_latent_axis=self.n_latent_axis)

        # Initialize Optimizer and Recommender based on one Mode
        if self.recommendation_type == RecommendationType.FUNCTION_BASED:
            self.recommender = BayesianRecommender(n_embedding_axis=self.n_embedding_axis,
                                                   n_latent_axis=self.n_latent_axis)
            self.optimizer = NoOptimizer()
            #TODO: Remove use of bound ares as they are not really variable anymore (fixed to [0., 1.])
        elif self.recommendation_type == RecommendationType.WEIGHTED_AXES:
            self.recommender = SinglePointWeightedAxesRecommender(n_embedding_axis=self.n_embedding_axis,
                                                                  n_latent_axis=self.n_latent_axis)
            self.optimizer = WeightedSumOptimizer()
        elif self.recommendation_type == RecommendationType.EMA_WEIGHTED_AXES:
            self.recommender = SinglePointWeightedAxesRecommender(n_embedding_axis=self.n_embedding_axis,
                                                                  n_latent_axis=self.n_latent_axis)
            self.optimizer = EMAWeightedSumOptimizer(n_recommendations=self.n_recommendations, alpha=self.ema_alpha)
        elif self.recommendation_type == RecommendationType.RANDOM:
            self.recommender = RandomRecommender(n_embedding_axis=self.n_embedding_axis,
                                                 n_latent_axis=self.n_latent_axis)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.EMA_DIRICHLET:
            self.recommender = DirichletRecommender(n_embedding_axis=self.n_embedding_axis,
                                                    n_latent_axis=self.n_latent_axis)
            self.optimizer = EMAWeightedSumOptimizer(n_recommendations=self.n_recommendations, alpha=self.ema_alpha)
        else:
            raise ValueError(f"The recommendation type {self.recommendation_type} is not implemented yet.")

    def inv_transform(self, user_embeddings: Tensor):
        """
        This function transforms embeddings in the user_space back into the clip embedding space.

        Parameters:
            user_embeddings (Tensor): Parameters concerning the initially defined axis of a user_embedding.

        Returns
            clip_embeddings (Tensor): The respective clip embeddings.
        """
        if self.n_latent_axis:
            latent_factors = user_embeddings[:, -self.latent_axis.shape[0]:]
            user_embeddings = user_embeddings[:, :-self.latent_axis.shape[0]]

        # r = n_rec, a = n_axis, t = n_tokens, e = embedding_size
        product = torch.einsum('ra,ate->rte', user_embeddings, self.embedding_axis)
        embedding_length = self.embedding_length.reshape((1, product.shape[1], 1))
        clip_embeddings = (self.embedding_center + product)
        clip_embeddings = (clip_embeddings / torch.linalg.vector_norm(clip_embeddings, ord=2, dim=-1, keepdim=True)
                           * embedding_length)

        latents = None
        if self.n_latent_axis:
            latents = self.latent_center + torch.einsum('rl,lxyz->rxyz', latent_factors, self.latent_axis)
            latents = torch.nan_to_num(latents, nan=0.0)  # avoid SVD LinAlgError for all zero preferences
            latents = (latents / torch.linalg.matrix_norm(latents, ord=2, dim=(-2, -1), keepdim=True)
                       * self.latent_space_length)

        return clip_embeddings, latents

    def fit_user_profile(self, preferences: Tensor):
        """
        This function initializes and fits a gaussian process for the available user preferences that can subsequently
        be used to generate new interesting embeddings for the user.

        Parameters:
            preferences (Tensor) : Preferences regarding the embeddings recommended last as real valued numbers.
        Returns:
            user_profile (Variable) : The fitted user profile depending on the optimizer.
        """
        # Initialize or extend the available user related data 
        if self.preferences is None:
            self.preferences = preferences
        else:
            self.preferences = torch.cat((self.preferences, preferences))
        
        # Only fit user profile if preferences are not all zero
        if torch.count_nonzero(self.preferences) > 0:
            self.user_profile = self.optimizer.optimize_user_profile(self.embeddings, self.preferences, self.user_profile)

    def clip_embedding(self, prompt: str):
        """
        Embeds a given prompt using CLIP.

        Returns:
            embedding (Tensor) : An embedding for the prompt in shape (1, 77, 768)
        """
        prompt_tokens = self.tokenizer(prompt,
                                       padding="max_length",
                                       max_length=self.tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt", ).to(self.text_encoder.device)

        prompt_embeds = self.text_encoder(prompt_tokens.input_ids)[0].cpu()
        return prompt_embeds.reshape(self.n_clip_tokens, self.embedding_dim)

    def generate_recommendations(self, num_recommendations: int = 2):
        """
        This function generates recommendations based on the previously fit user-profile.

        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Trade-off between exploration and exploitation.
                Must be in [0, 1]. 0 means exploration, 1 means exploitation.
                Beta is increased after each recommendation (i.e. more exploitation).
                Optional, if given (by the debug menu), it will be used for the next generation of images.
        Returns:
            embeddings (Tensor): Embeddings that can be retransformed into the CLIP space and used for image generation
        """
        # Generate recommendations in the user_space
        if self.user_profile is not None:
            # obtain beta from the recommender if not given
            user_space_embeddings = self.recommender.recommend_embeddings(user_profile=self.user_profile,
                                                                          n_recommendations=num_recommendations//2,
                                                                          beta=self.beta)
            
            # Include some random user_space_embeddings througout each iteration
            random_user_space_embeddings = self.random_recommender.recommend_embeddings(None, num_recommendations//2)
            user_space_embeddings = torch.cat((user_space_embeddings, random_user_space_embeddings))

            # Update Beta
            self.beta = min(self.beta+self.beta_step_size, 1.)
        else:
            # Start initially with a lot of random embeddings to build a foundation for the user profile
            user_space_embeddings = self.random_recommender.recommend_embeddings(None, num_recommendations)

        # Safe the user_space_embeddings
        if self.embeddings is not None:
            self.embeddings = torch.cat((self.embeddings, user_space_embeddings))
        else:
            self.embeddings = user_space_embeddings

        # Transform embeddings from user_space to CLIP space
        clip_embeddings, latents = self.inv_transform(user_space_embeddings)
        return clip_embeddings, latents
    
    def generate_image_grid(self):
        """
        This function creates a set of user embeddings for the creation of the image wall. In general, the 
        user profile in form of a weighted center is approximatly in the middle.
        Returns:
            Meshgrid (Tensor) : A meshgrid of samples going from lower left to upper right column-wise. So (-1, -1)
                (-1, -0.677), (-1, -0.5), ...
        """
        # Calculate the PCA for current embeddings 
        matrix = self.embeddings
        pca = PCA(n_components=2).fit(matrix)

        # Create a meshgrid in the 2D space
        grid_x = torch.linspace(-1, 1, 7)
        grid_y = torch.linspace(-1, 1, 7)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_xy = torch.cat((grid_x.flatten().reshape(-1, 1), grid_y.flatten().reshape(-1, 1)), dim=1)

        # Retransform back into User-Space
        grid_xy_re = pca.inverse_transform(grid_xy)

        # Return the grid to be plottet
        return grid_xy_re

    def plotting_utils(self, algorithm: str = 'pca'):
        # TODO: Discuss removing tsne as it has no backwards compatibility
        """
        This function creates a reduction of the user embeddings into a two-dimensional space, so we can plot the
        embedding space and the respective images in our application.
        Parameters:
            algorithm (str) : Defines, which algorithm to use for the reduction.
        Returns:
            2D-user_profile (Tensor) : The user profile on which we base our recommendations on.
            2D-user_embeddings (Tensor) : Two dimensional reduction of the embeddings that resulted in the previously
                generated images
            Preferences (Tensor) : The respective preferences as a number between 0 and 1.
        """

        if self.num_axis == 2:
            return self.user_profile, self.embeddings, self.preferences

        else:
            # Check for GP-User Embedding
            if self.recommendation_type == RecommendationType.FUNCTION_BASED or self.recommendation_type == RecommendationType.RANDOM:
                matrix = self.embeddings
                pca = PCA(n_components=2).fit(matrix)
                transformed_embeddings = pca.transform(matrix)

                if self.recommendation_type == RecommendationType.RANDOM:
                    return None, transformed_embeddings, self.preferences

                # Retrieve scores for heatmap (function-based recommender)
                grid_x = torch.linspace(-1, 1, 200)
                grid_y = torch.linspace(-1, 1, 200)
                grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
                low_d_user_space = torch.cat((grid_x.flatten().reshape(-1, 1), grid_y.flatten().reshape(-1, 1)), dim=1)
                user_space = pca.inverse_transform(low_d_user_space).float()
                scores = self.recommender.heat_map_values(user_profile=self.user_profile,
                                                          user_space=user_space)
                if scores is not None:
                    scores = scores.reshape(grid_x.shape)

                return (low_d_user_space[:,0], low_d_user_space[:,1], scores), transformed_embeddings, self.preferences

            else:
                matrix = torch.cat((self.user_profile.reshape(1, -1), self.embeddings), dim=0)
                if algorithm == 'pca':
                    pca = PCA(n_components=2)
                    transformed_embeddings = pca.fit_transform(matrix)
                elif algorithm == 'tsne':
                    transformed_embeddings = TSNE(random_state=42).fit_transform(matrix)
                else:
                    raise NotImplementedError(f'The requested reduction algorithm ({algorithm}) is not available.')

                low_d_user_profile = transformed_embeddings[0]
                low_d_embeddings = transformed_embeddings[1:]
                return low_d_user_profile, low_d_embeddings, self.preferences
