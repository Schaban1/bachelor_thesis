from abc import abstractmethod, ABC
from torch import Tensor

import torch
from PIL.Image import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


class GeneratorBase(ABC):

    @abstractmethod
    def __init__(self, n_images: int=5):
        pass

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass


class Generator(GeneratorBase):
    def __init__(self, n_images=5, hf_model_name: str="stable-diffusion-v1-5/stable-diffusion-v1-5", cache_dir: str|None='/cache/', 
                 num_inference_steps : int = 20, device : str = 'cuda', random_latents : bool = False):
        """
        Setting the image generation scheduler, SD pipeline, and latents that stay constant during the iterative refining.

        Args:
            n_images: the number of embeddings that will be generated in a batch and returned from generate_images
            hf_model_name: Huggingface model identifier, default is Stable diffusion 1.5
        """
        self.height = 512
        self.width = 512
        self.random_latents = random_latents
        self.num_inference_steps=num_inference_steps
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            steps_offset=1
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            scheduler=scheduler,
            safety_checker = None,
            requires_safety_checker = False,
            cache_dir=cache_dir,
        )
        self.device = torch.device("cuda") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
        self.pipe.to(self.device)
        self.n_images = n_images

        self.latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
            device=self.device,
        ).repeat(n_images, 1, 1, 1)


    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        """
        Generates a list of image(s) from given embedding

        Args:
        embedding (Tensor or tuple[Tensor, Tensor]):
            A single embedding as tensor of shape (batch, 77, 768)
            A tuple with two embedding tensors each with 3 dim (batch, 77, 768)

        Returns:
            `list[PIL.Image.Image]: a list of batch many PIL images generated from the embeddings.
        """
        if type(embedding) == tuple:
            latent = embedding[1].to(self.device)
            embedding = embedding[0].to(self.device)
        else:
            if self.random_latents:
                latent = torch.randn(
                (self.n_images, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
                device=self.device,
                )
            else:
                latent = self.latents

        return self.pipe(height=self.height,
            width=self.width,
            num_images_per_prompt=1,
            prompt_embeds=embedding,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=7,
            latents=latent,
        ).images