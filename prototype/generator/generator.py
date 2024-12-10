from abc import abstractmethod, ABC
from torch import Tensor

import torch
from PIL.Image import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


class GeneratorBase(ABC):

    @abstractmethod
    def __init__(self, n_images: int = 5):
        pass

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass


class Generator(GeneratorBase):
    def __init__(self,
                 n_images=5,
                 hf_model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 cache_dir: str | None = '/cache/',
                 num_inference_steps: int = 20,
                 batch_size: int | None = None, ):
        """
        Setting the image generation scheduler, SD pipeline, and latents that stay constant during the iterative refining.

        Args:
            n_images: the number of embeddings that will be generated in a batch and returned from generate_images
            hf_model_name: Huggingface model identifier, default is Stable diffusion 1.5
            cache_dir: directory to download to model to
            num_inference_steps: number of denoising steps for the model to take
            batch_size: number of images that should be generated in a batch, lower means less vram needed
        """
        self.height = 512
        self.width = 512
        self.num_inference_steps = num_inference_steps
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
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=cache_dir,
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pipe.to(self.device)
        self.n_images = n_images
        self.batch_size = batch_size

        self.latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
            device=self.device,
        ).repeat(n_images, 1, 1, 1)

    def generate_image(self, embeddings: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        """
        Generates a list of image(s) from given embedding

        Args:
        embedding (Tensor or tuple[Tensor, Tensor]):
            A single embedding as tensor of shape (batch, 77, 768)
            A tuple with two embedding tensors each with 3 dim (batch, 77, 768)

        Returns:
            `list[PIL.Image.Image]: a list of batch many PIL images generated from the embeddings.
        """
        pos_prompt_embeds = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        neg_prompt_embeds = embeddings[1] if isinstance(embeddings, tuple) else None
        num_embeddings = pos_prompt_embeds.shape[0]
        batch_steps = self.batch_size or num_embeddings

        images = []
        for i in range(0, num_embeddings, batch_steps):
            images.extend(self.pipe(height=self.height,
                                    width=self.width,
                                    num_images_per_prompt=1,
                                    prompt_embeds=pos_prompt_embeds[i:i + batch_steps],
                                    negative_prompt_embeds=neg_prompt_embeds[i:i + batch_steps] if neg_prompt_embeds else None,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=7,
                                    latents=self.latents[i:i + batch_steps],
                                    ).images
                          )
        return images


if __name__ == "__main__":
    n_images = 3
    gen = Generator(n_images=n_images, batch_size=1, cache_dir=None, num_inference_steps=25)
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    prompt_tokens = gen.pipe.tokenizer(prompt,
                                       padding="max_length",
                                       max_length=gen.pipe.tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt",
                                       )
    embed = gen.pipe.text_encoder(prompt_tokens.input_ids.to(gen.device))[0]
    embed = embed.repeat(n_images,1,1)
    print(f"{embed.shape=}")
    img = gen.generate_image(embed)
    for i in range(n_images):
        img[i].save(f"../output/{i}.png")

    print(img)
