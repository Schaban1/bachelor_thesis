from abc import abstractmethod, ABC
from torch import Tensor

import torch
from PIL.Image import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


class GeneratorBase(ABC):

    @abstractmethod
    def embed_prompt(self, prompt: str, negative_prompt: str = None) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass


class Generator(GeneratorBase):
    def __init__(self, hf_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5", cache_dir='/cache/'):
        self.height = 512
        self.width = 512
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            steps_offset=1,
        )
        #self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            scheduler=scheduler,
            safety_checker = None,
            requires_safety_checker = False,
            cache_dir=cache_dir
        )
        self.device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pipe.to(self.device)

    # TODO deprecated
    def embed_prompt(self, prompt: str, negative_prompt: str = None) -> tuple[Tensor, Tensor]:
        """
        Embeds a given prompt and a negative prompt

        Returns:
            `Tuple[Tensor, Tensor]: A tuple of embeddings for the prompt and negative prompt in shape (1, 77, 768)
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
        batch_size = embedding.shape[0]
        # TODO (Discuss Paul): Shouldnt we keep the latent constant for all prompts to avoide additional factors impacting the generation?
        # Proposition: Create a original latent when the generator is generated and then use expand here to expand it to the batch size.
        latents = torch.randn(
            (batch_size, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
            device=self.device
        )

        if type(embedding) != tuple:
            return self.pipe(height=self.height,
                width=self.width,
                num_images_per_prompt=1,
                prompt_embeds=embedding,
                num_inference_steps=20,
                guidance_scale=7,
                latents=latents,
            ).images

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

