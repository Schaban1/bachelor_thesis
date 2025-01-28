from abc import abstractmethod, ABC
from torch import Tensor

from nicegui import binding
import torch
from PIL.Image import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


class GeneratorBase(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass


class Generator(GeneratorBase):
    height = binding.BindableProperty()
    width = binding.BindableProperty()
    batch_size = binding.BindableProperty()
    random_latents = binding.BindableProperty()
    num_inference_steps = binding.BindableProperty()
    guidance_scale = binding.BindableProperty()
    n_images = binding.BindableProperty()
    use_negative_prompt = binding.BindableProperty()

    def __init__(self,
                 n_images=5,
                 batch_size=None,
                 hf_model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 cache_dir: str | None = '/cache/',
                 num_inference_steps: int = 20,
                 device: str = 'cuda',
                 random_latents: bool = False,
                 guidance_scale: float = 7.,
                 use_negative_prompt: bool = False
                 ):
        """
        Setting the image generation scheduler, SD pipeline, and latents that stay constant during the iterative refining.

        Args:
            n_images: the number of embeddings that will be generated in a batch and returned from generate_images
            hf_model_name: Huggingface model identifier, default is Stable diffusion 1.5
            cache_dir: directory to download to model to
            num_inference_steps: number of denoising steps for the model to take
            batch_size: number of images that should be generated in a batch, lower means less vram needed
            device: gpu or cpu that should be used to generate images
        """

        self.height = 512
        self.width = 512
        self.batch_size = batch_size
        self.random_latents = random_latents
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.n_images = n_images
        self.use_negative_prompt = use_negative_prompt
        self.latest_images = []

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
            #torch_dtype=torch.float16
        )

        self.device = torch.device("cuda") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=cache_dir ,torch_dtype=torch.float16).to(self.device)
        # self.pipe.vae = vae

        self.pipe.to(self.device)
        #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.load_generator()
    
    def load_generator(self):
        self.latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
            device=self.device,# dtype=torch.float16
        ).repeat(self.n_images, 1, 1, 1)

        self.negative_prompt_embeds = None
        if self.use_negative_prompt:
            negative_prompt = "lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck, illustration, painting, drawing, art, sketch,bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, worst quality, cropped, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, deformed, missing limb, bad hands, extra digits, extra fingers, not enough fingers, floating head, disembodied"
            negative_prompt_tokens = self.pipe.tokenizer(negative_prompt,
                                                         padding="max_length",
                                                         max_length=self.pipe.tokenizer.model_max_length,
                                                         truncation=True,
                                                         return_tensors="pt", ).to(self.pipe.text_encoder.device)
            self.negative_prompt_embeds = self.pipe.text_encoder(negative_prompt_tokens.input_ids)[0].repeat(self.n_images, 1, 1)

    def generate_image(self, embeddings: Tensor | tuple[Tensor, Tensor], latents: Tensor = None) -> list[Image]:
        """
        Generates a list of image(s) from given embedding

        Args:
        embedding (Tensor or tuple[Tensor, Tensor]):
            A single embedding as tensor of shape (batch, 77, 768)
            A tuple with two embedding tensors each with 3 dim (batch, 77, 768)

        Returns:
            `list[PIL.Image.Image]: a list of batch many PIL images generated from the embeddings.
        """
        # if embeddings.dtype == torch.float32:
        #     embeddings = embeddings.type(torch.float16)

        embeddings = embeddings.to(self.device)
        if latents != None:
            latents = latents.to(self.device)
            #latents = latents.type(torch.float16)
        else:
            if self.random_latents:
                latents = torch.randn(
                    (self.n_images, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
                    device=self.device,# dtype=torch.float16
                )
            else:
                latents = self.latents

        pos_prompt_embeds = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        neg_prompt_embeds = embeddings[1] if isinstance(embeddings, tuple) else self.negative_prompt_embeds
        num_embeddings = pos_prompt_embeds.shape[0]
        batch_steps = self.batch_size or num_embeddings

        images = []
        for i in range(0, num_embeddings, batch_steps):
            images.extend(self.pipe(height=self.height,
                                    width=self.width,
                                    num_images_per_prompt=1,
                                    prompt_embeds=pos_prompt_embeds[i:i + batch_steps],
                                    negative_prompt_embeds=neg_prompt_embeds[i:i + batch_steps] if neg_prompt_embeds is not None else None,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=self.guidance_scale,
                                    latents=latents[i:i + batch_steps],
                                    ).images
                          )

        self.latest_images.extend(images)
        return images

    def get_latest_images(self):
        """
        Returns the latest generated images in the "cache" and clears the cache.
        This is useful to remove already displayed images from the memory.
        """
        latest_images = self.latest_images
        self.latest_images = []
        return latest_images

    def clear_latest_images(self):
        self.latest_images = []


if __name__ == "__main__":
    n_images = 3
    gen = Generator(n_images=n_images,
                    batch_size=None,
                    cache_dir=None,
                    num_inference_steps=25,
                    use_negative_prompt=False,
                    random_latents=True)
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    prompt_tokens = gen.pipe.tokenizer(prompt,
                                       padding="max_length",
                                       max_length=gen.pipe.tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt",
                                       )

    embed = gen.pipe.text_encoder(prompt_tokens.input_ids.to(gen.device))[0]
    embed = embed.repeat(n_images, 1, 1)
    print(f"{embed.shape=}")
    for i in range(5):
        img = gen.generate_image(embed)
        for i in range(n_images):
            img[i].save(f"../output/{i}.png")

    print(img)
