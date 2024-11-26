from abc import abstractmethod, ABC
from torch import Tensor

import torch
from PIL.Image import Image
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, LMSDiscreteScheduler


class GeneratorBase(ABC):

    @abstractmethod
    def embed_prompt(self, prompt: str | list[str]) -> Tensor:
        pass

    @abstractmethod
    def generate_image(self, embedding: Tensor | list[Tensor]) -> Image:
        pass


class Generator(GeneratorBase):
    def __init__(self, mock=False):
        self.mock = mock  # TODO temporary
        self.height = 512
        self.width = 512
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            steps_offset=1
        )
        #self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            scheduler=scheduler,
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pipe.to(self.device)

    def embed_prompt(self, prompt: str, negative_prompt: str | list[str] = None) -> tuple[Tensor, Tensor]:
        """
        Embeds a given prompt and a negative prompt

        Returns:
            `Tuple[Tensor, Tensor]: A tuple of embeddings for the prompt and negative prompt in shape (1, ?, 768)
        """

        prompt_tokens = self.pipe.tokenizer(prompt,
                            padding="max_length",
                            max_length=self.pipe.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",)

        prompt_embeds = self.pipe.text_encoder(prompt_tokens.input_ids.to(self.device))[0]
        if negative_prompt is None:
            negative_prompt = [""]

        negative_prompt_tokens =self. pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = self.pipe.text_encoder(negative_prompt_tokens.input_ids.to(self.device))[0]

        return prompt_embeds, negative_prompt_embeds


    def generate_image(self, embedding: tuple[Tensor, Tensor]) -> list[Image]:
        """
        Generates a list of image(s) from given embedding

        Args:
        embedding (tuple[Tensor, Tensor]):
            A tuple with two Tensors each with 3 dim (batch, ?, 768)

        Returns:
            `List[PIL.Image.Image]: a list of batch many PIL images generated from the embeddings.
        """
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
            device=self.device
        )

        return self.pipe(height=self.height,
            width=self.width,
            num_images_per_prompt=1,
            prompt_embeds=embedding[0],
            negative_prompt_embeds=embedding[1],
            num_inference_steps=20,
            guidance_scale=7,
            latents=latents,
        ).images



if __name__ == "__main__":
    gen = Generator()
    embed = gen.embed_prompt("A cinematic shot of a baby racoon wearing an intricate italian priest robe.")
    print(embed[0].shape)
    print(embed[1].shape)
    img = gen.generate_image(embed)
    print(img)

