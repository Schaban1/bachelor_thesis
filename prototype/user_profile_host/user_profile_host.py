import json
import random
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from nicegui import binding

from .recommender import *
from .optimizer import *
from ..constants import RecommendationType
from diffusers import StableDiffusionPipeline


class UserProfileHost():
    original_prompt = binding.BindableProperty()
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
    include_random_rec = binding.BindableProperty()

    # TODO: Group together Recommender Args and just pass them to the recommender, should simplyfy this arg list
    def __init__(
            self,
            original_prompt: str,
            add_ons: list = None,
            recommendation_type: str = RecommendationType.RANDOM,
            stable_dif_pipe: StableDiffusionPipeline = None,
            hf_model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir: str = './cache/',
            n_embedding_axis: int = 13,
            use_embedding_center: bool = True,
            n_latent_axis: int = 3,
            use_latent_center: bool = False,
            n_recommendations: int = 6,
            include_random_recommendations: bool = False,
            ema_alpha: float = 0.5,
            beta: float = 0.3,
            beta_step_size: float = 0.1,
            axis_style: str = 'ordered'
    ):
        """
        This class is the main interface for the user profile host. It initializes the user profile host with the
        :param original_prompt: The original prompt as string.
        :param add_ons: A list of additional prompts to be used as axis for the user profile space.
            Elements of the list are strings.
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
        :param beta: Trade-off between exploration and exploitation. Must be in [0, 1]. 0 means exploration, 1 means
            exploitation. Beta is increased after each recommendation (i.e. more exploitation).
        :param beta_step_size: The step size for the beta increase.
        """
        # Some Clip Hyperparameters
        self.original_prompt = original_prompt
        self.add_ons = add_ons
        self.recommendation_type = recommendation_type
        self.stable_dif_pipe = stable_dif_pipe
        self.embedding_dim = 768
        self.n_clip_tokens = 77
        self.height = 512
        self.width = 512
        self.latent_space_length = 15.55
        self.n_latent_axis = (n_latent_axis * 2) if self.recommendation_type == RecommendationType.SIMPLE else n_latent_axis
        self.n_embedding_axis = n_embedding_axis
        self.use_embedding_center = use_embedding_center
        self.use_latent_center = use_latent_center
        self.n_recommendations = n_recommendations
        self.ema_alpha = ema_alpha
        self.beta = min(beta, 1.)
        self.beta_step_size = beta_step_size
        self.include_random_rec = include_random_recommendations
        self.axis_style = axis_style

        # Check for valid values
        assert self.beta >= 0., "Beta should be in range [0., 1.]"
        assert self.beta_step_size >= 0. and self.beta_step_size < 1., "Beta Step Size should be in [0., 1.]"

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = torch.tensor([])

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Holds previous low dimensional user profiles
        self.user_profile_history = []

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
        self.prompt_embedding = self.clip_embedding(self.original_prompt)
        self.embedding_length = torch.linalg.vector_norm(self.prompt_embedding, ord=2, dim=-1, keepdim=False)
        if not self.use_embedding_center:
            self.embedding_center = torch.zeros(size=(1, self.n_clip_tokens, self.embedding_dim))
        else:
            self.embedding_center = self.prompt_embedding

        # Generate axis to define the user profile space with extensions of the original user-promt in the clip embedding space
        with open('prototype/user_profile_host/prompt_terms.json', 'r') as f:
            prompt_terms = json.load(f)

        self.image_styles = prompt_terms["image_styles"] 
        self.secondary_contexts = prompt_terms["secondary_contexts"]
        self.atmospheric_attributes = prompt_terms["atmospheric_attributes"]
        self.quality_terms = prompt_terms["quality_terms"]

        # Create Add ons with original prompt included at the semantically correct position
        if self.axis_style == 'ordered':
            self.add_ons = []
            for _ in range(self.n_embedding_axis):
                ao = random.choice(self.image_styles) + self.original_prompt + random.choice(self.secondary_contexts) + random.choice(self.atmospheric_attributes) + random.choice(self.quality_terms)
                self.add_ons.append(ao)
        elif self.axis_style == 'random':
            self.add_ons = [
                "A beautiful purple flower in a dark forest, in the style of hyper-realistic sculptures, with dark orange and green colors, set against post-apocalyptic backdrops with light red and yellow hues. it is displayed in museum gallery dioramas, featuring soft, dreamy scenes with an orange and green surreal 8k zbrush render.",
                "Fluid abstract background, dark indigo, art, behance",
                "hyperdetailed eyes, tee-shirt design, line art, black background, ultra detailed artistic, detailed gorgeous face, natural skin, water splash, colour splash art, fire and ice, splatter, black ink, liquid melting, dreamy, glowing, glamour, glimmer, shadows, oil on canvas, brush strokes, smooth, ultra high definition, 8k, unreal engine 5, ultra sharp focus, intricate artwork masterpiece, ominous, golden ratio, highly detailed, vibrant, production cinematic character render, ultra high quality model",
                "Futuristic sci-fi pod chair, flat design, product-view, editorial photography",
                "Cute girl behind window, rainy, photography surreal art, blurry, minimalistic",
                "Shadowy figure of a woman emerging from the darkness, black and grey gradient, foggy, realistic, 8k resolution, unreal engine, cinematic",
                "Old man standing next to a giant monster, in the style of contemporary vintage photography, necronomicon illustrations, tabletop photography, 1890, hyperrealistic animal portraits, ghostly presence, whirring contrivances",
                "Victo ngai style",
                "Detailed, vibrant illustration of a cowboy in the copper canyons the sierra of chihuahua state, by herge, in the style of tin-tin comics, vibrant colors, detailed, sunny day, attention to detail, 8k",
                "Create a surreal desert with alien plants, the plants are shaped like canary_yellow_perlwhite, are partially transparent with tentacles and spines, in the sand laying pearls, backdrop is the storm of cosmic dust and cosmic clouds the heaven is dark colored unreal engine 6 color palette knives painting oel on canvas conzeptart, high qualty, cinema_stil, wide shot",
                "beautiful field of flowers, colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain",
                "A boy playing video games at night in his room, illustration by hergé, perfect coloring, 8k",
                "Drawing of a cosmic extraterrestrial technology healing chamber, with many cables connecting the chamber to a large translucent transparent crystal. a body silhouette inside. ambient aircraft.",
                "An intricate village made of psychedelic mushrooms, art by greg rutkowsk, 3d render",
                "Some people look over tall building windows, in the style of dark hues, rural china, coded patterns, sparse and simple, uhd image, urbancore, sovietwave, negative space, award-winning design",
                "Diesel-punk hip-hop punk ashigaru wearing diesel-punk oni armor. full body fighting pose. traditional wet ink and watercolor painting style. black, grey, red, and metallic gold ink. gestural speed paint by artgerm and jungshan. street fighter style.",
                "A cute minimalistic simple capybara side profile, in the style of jon klassen, desaturated light and airy pastel color palette, nursery art, white background",
                "black and red ink, a crane in chinese style, ink art by mschiffer, whimsical, rough sketch, (sketch1.3)",
                "A cute cartoon girl in a dress holding a white kitten, full body, yellow background, keith haring style doodle, sharpie illustration, bold lines and solid colors, simple details, (((minimalism))), yellow background",
                "Japanese animation, panoramic, colorful, a small corgi with closed eyes backstroke in the pool, most of the picture shows water, corgi accounts for a small part of the picture, water is light blue transparent and clear, water ripple texture is clear, light refraction, corgi and water are not fuzzy, in hd, phone wallpaper size, hd, 32k",
                "Body portrait photography, in a smoke-filled office full of cables and wires and led, featuring a carbon motor head, an attractive transparent white plexiglass secretary robot reading an ancient book at her desk, 80-degree view. art by sergio lopez, natalie shau, james jean, and salvador dali."
            ][:self.n_embedding_axis]
            # Include original prompt if not using the embedding center to remain the primary context
            if not self.use_embedding_center:
                self.add_ons = [prompt + ', ' + a for a in self.add_ons]
        else:
            raise NotImplementedError()
  
        self.embedding_axis = []
        print('The embedding axis will consist of the following prompts:')
        for prompt in self.add_ons:
            print(prompt)
            self.embedding_axis.append(self.clip_embedding(prompt))
        self.embedding_axis = torch.stack(self.embedding_axis)

        # Similarly, define axis in the latent space to have variations in both spaces that together build the user space
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

        # Generally required
        self.random_recommender = RandomRecommender(n_embedding_axis=self.n_embedding_axis, n_latent_axis=self.n_latent_axis)

        # Initialize Optimizer and Recommender based on one Mode
        if self.recommendation_type == RecommendationType.FUNCTION_BASED:
            self.recommender = BayesianRecommender(n_embedding_axis=self.n_embedding_axis,
                                                   n_latent_axis=self.n_latent_axis)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.RANDOM:
            self.recommender = RandomRecommender(n_embedding_axis=self.n_embedding_axis,
                                                 n_latent_axis=self.n_latent_axis)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.EMA_DIRICHLET:
            self.recommender = DirichletRecommender(n_embedding_axis=self.n_embedding_axis,
                                                    n_latent_axis=self.n_latent_axis)
            self.optimizer = EMAWeightedSumOptimizer(n_recommendations=self.n_recommendations, alpha=self.ema_alpha)
        elif self.recommendation_type == RecommendationType.BASELINE:
            self.recommender = BaselineRecommender(n_latent_axis=self.n_latent_axis)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.SIMPLE:
            self.recommender = SimpleRandomRecommender(n_embedding_axis=self.n_embedding_axis,
                                                    n_latent_axis=self.n_latent_axis)
            self.optimizer = SimpleOptimizer(n_embedding_axis=self.n_embedding_axis,
                                             n_latent_axis=self.n_latent_axis,
                                             image_styles=self.image_styles,
                                             secondary_contexts=self.secondary_contexts,
                                             atmospheric_attributes=self.atmospheric_attributes,
                                             quality_terms=self.quality_terms)
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
        if not self.recommendation_type == RecommendationType.BASELINE:
            user_embeddings = user_embeddings.type(self.text_encoder.dtype)
            self.embedding_axis = self.embedding_axis.type(self.text_encoder.dtype)
            product = torch.einsum('ra,ate->rte', user_embeddings, self.embedding_axis)
            embedding_length = self.embedding_length.reshape((1, product.shape[1], 1))
            clip_embeddings = (self.embedding_center + product)
            clip_embeddings = (clip_embeddings / torch.linalg.vector_norm(clip_embeddings, ord=2, dim=-1, keepdim=True)
                            * embedding_length)
        else:
            clip_embeddings = self.prompt_embedding.repeat(user_embeddings.shape[0], 1, 1)

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
        if self.preferences is not None:
            self.preferences = torch.cat((self.preferences, preferences))
        else:
            self.preferences = preferences
            
        # Only fit user profile if preferences are not all zero
        if torch.count_nonzero(self.preferences) > 0:
            if self.user_profile is not None:
                self.user_profile_history.append(self.user_profile)
            self.user_profile = self.optimizer.optimize_user_profile(self.embeddings, self.preferences, self.user_profile, self.beta)


    @torch.no_grad()
    def clip_embedding(self, prompt: str):
        """
        Embeds a given prompt using CLIP.

        Returns:
            embedding (Tensor) : An embedding for the prompt in shape (77, 768)
        """
        prompt_embeds = self.stable_dif_pipe.encode_prompt(prompt,
                                                          device=self.text_encoder.device,
                                                          num_images_per_prompt=1,
                                                          do_classifier_free_guidance=False)[0].cpu()

        return prompt_embeds.squeeze()


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
        # The first recommender does not make use of the originally created subspace but dynamically generates new prompts based on user voting behavior
        if self.recommendation_type == RecommendationType.SIMPLE:
            # Extract probability distributions over all terms from user profile (None equals uniform distribution)
            if self.user_profile is not None:
                img_weights, sec_weights, at_weights, qual_weights, lat_weights = self.user_profile

                # Debug prints
                for weights, terms, name in zip([img_weights, sec_weights, at_weights, qual_weights], [self.image_styles, self.secondary_contexts, self.atmospheric_attributes, self.quality_terms], ['Image Styles:', 'Secondary Contexts:', 'Atmospheric Attributes:', 'Quality Terms:']):
                    top_val, top_idx = torch.topk(torch.tensor([w for w in weights]), k=4)
                    print("Top 4 "+name)
                    for val, idx in zip(top_val, top_idx):
                        print(terms[idx],'['+str(round(val.item()*100, 2))+'%]')
            else:
                img_weights, sec_weights, at_weights, qual_weights, lat_weights = None, None, None, None, None

            # Select new indices that build up the next embeddings
            img_idx = random.choices(range(len(self.image_styles)), weights=img_weights, k=num_recommendations)
            sec_idx = random.choices(range(len(self.secondary_contexts)), weights=sec_weights, k=num_recommendations)
            at_idx = random.choices(range(len(self.atmospheric_attributes)), weights=at_weights, k=num_recommendations)
            qual_idx = random.choices(range(len(self.quality_terms)), weights=qual_weights, k=num_recommendations)
            lat_idx = random.choices(range(self.n_latent_axis), weights=lat_weights, k=num_recommendations)

            # Generate respective clip embeddings (note that no inv-transformation is required here)
            print("The following prompts will be generated with various latents:")
            clip_embeddings = []
            for i in range(num_recommendations):
                prompt = self.image_styles[img_idx[i]] + self.original_prompt + self.secondary_contexts[sec_idx[i]] + self.atmospheric_attributes[at_idx[i]] + self.quality_terms[qual_idx[i]]
                print(str(i+1)+":", prompt)
                c_emb = self.clip_embedding(prompt)
                clip_embeddings.append(c_emb)
            clip_embeddings = torch.stack(clip_embeddings)

            # For latents, simply select the respective latent from a list
            latents = self.latent_axis[lat_idx]
            
            # Update user profile
            if self.embeddings is not None:
                self.embeddings[0].extend(img_idx)
                self.embeddings[1].extend(sec_idx)
                self.embeddings[2].extend(at_idx)
                self.embeddings[3].extend(qual_idx)
                self.embeddings[4].extend(lat_idx)
            else:
                self.embeddings = [img_idx, sec_idx, at_idx, qual_idx, lat_idx]

        # This case works with the predefined user axis
        else:
            # Generate recommendations in the user_space
            if self.user_profile is not None or self.recommendation_type == RecommendationType.BASELINE:
                # obtain beta from the recommender if not given
                user_space_embeddings = self.recommender.recommend_embeddings(user_profile=self.user_profile,
                                                                            n_recommendations=num_recommendations,
                                                                            beta=self.beta)
            else:
                # Start initially with a lot of random embeddings to build a foundation for the user profile
                user_space_embeddings = self.random_recommender.recommend_embeddings(None, num_recommendations)

            # Transform embeddings from user_space to CLIP space
            clip_embeddings, latents = self.inv_transform(user_space_embeddings)

            user_space_embeddings.type(self.text_encoder.dtype)
            # Safe the user_space_embeddings
            if self.embeddings is not None:
                self.embeddings = torch.cat((self.embeddings, user_space_embeddings))
            else:
                self.embeddings = user_space_embeddings

        # Update Beta and return clip embeddings and latents for a generator to use
        self.beta = min(self.beta+self.beta_step_size, 1.)
        return clip_embeddings, latents


    def plotting_utils(self):
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

        assert self.recommendation_type != RecommendationType.SIMPLE, "This is not yet available for the simple recommender."

        if self.num_axis == 2:
            return self.user_profile, self.embeddings, self.preferences

        else:
            # Check for GP-User Embedding
            if self.recommendation_type == RecommendationType.FUNCTION_BASED or self.recommendation_type == RecommendationType.RANDOM or self.recommendation_type == RecommendationType.DIVERSE_DIRICHLET:
                matrix = self.embeddings
                pca = PCA(n_components=2).fit(matrix)
                transformed_embeddings = pca.transform(matrix)

                if self.recommendation_type == RecommendationType.RANDOM or self.recommendation_type == RecommendationType.DIVERSE_DIRICHLET:
                    return None, transformed_embeddings, self.preferences

                # Retrieve scores for heatmap (function-based recommender)
                x = torch.linspace(-1, 1, 200)
                y = torch.linspace(-1, 1, 200)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                low_d_user_space = torch.cat((grid_x.flatten().reshape(-1, 1), grid_y.flatten().reshape(-1, 1)), dim=1)
                user_space = pca.inverse_transform(low_d_user_space).type(self.text_encoder.dtype)
                scores = self.recommender.heat_map_values(user_profile=self.user_profile,
                                                          user_space=user_space)
                if scores is not None:
                    scores = scores.reshape(grid_x.shape)

                return (x, y, scores), transformed_embeddings, self.preferences

            else:
                # First iteration, no user profile yet
                if self.user_profile is None:
                    matrix = self.embeddings

                else:
                    matrix = torch.cat((self.user_profile.reshape(1, -1), self.embeddings), dim=0)

                pca = PCA(n_components=2)
                transformed_embeddings = pca.fit_transform(matrix)

                if self.user_profile is None:
                    return None, transformed_embeddings, self.preferences

                else:
                    print(f'User profile history: {self.user_profile_history}')
                    low_d_user_profile = transformed_embeddings[0]
                    low_d_embeddings = transformed_embeddings[1:]
                    return low_d_user_profile, low_d_embeddings, self.preferences
