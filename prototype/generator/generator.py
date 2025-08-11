import logging
import time
import torch
from PIL.Image import Image
from abc import abstractmethod, ABC
from diffusers import StableDiffusionPipeline, AutoencoderTiny, AutoencoderKL, LCMScheduler
from functools import partial
from nicegui import binding
from torch import Tensor



class GeneratorBase(ABC):

    def __init__(self):
        self.latest_images = []

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass

    def get_latest_images(self) -> list[Image]:
        """
        Returns the latest generated images in the "cache" and clears the cache.
        This is useful to remove already displayed images from the memory.
        """
        latest_images = self.latest_images
        self.latest_images = []
        return latest_images

    def clear_latest_images(self) -> None:
        self.latest_images = []


class Generator(GeneratorBase):
    height = binding.BindableProperty()
    width = binding.BindableProperty()
    batch_size = binding.BindableProperty()
    num_inference_steps = binding.BindableProperty()
    guidance_scale = binding.BindableProperty()
    use_negative_prompt = binding.BindableProperty()

    @torch.no_grad()
    def __init__(self,
                 batch_size: int = None,
                 hf_model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 cache_dir: str | None = '/cache/',
                 num_inference_steps: int = 20,
                 device: str = 'cuda',
                 guidance_scale: float = 7.,
                 use_negative_prompt: bool = False,
                 callback=None,
                 pipe=None,
                 initial_latent_seed: int = 42
                 # todo this is unused here, but currently passed from config.yaml. should be removed
                 ):
        """
        Setting the image generation scheduler, SD pipeline, and latents that stay constant during the iterative refining.

        Args:
            hf_model_name: Huggingface model identifier, default is Stable diffusion 1.5
            cache_dir: directory to download to model to
            num_inference_steps: number of denoising steps for the model to take
            batch_size: number of images that should be generated in a batch, lower means less vram needed
            device: gpu or cpu that should be used to generate images
        """
        super().__init__()
        self.height = 512
        self.width = 512
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_negative_prompt = use_negative_prompt
        self.callback = callback

        self.device = torch.device("cuda") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        self.initial_latent_generator = torch.Generator(device=self.device)
        self.initial_latent_seed = initial_latent_seed
        self.initial_latent_generator.manual_seed(self.initial_latent_seed)

        self.pipe = pipe if pipe else StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            #safety_checker=None,
            requires_safety_checker=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdv1-5"
        )
        pipe.fuse_lora()

        # self.pipe.unet = torch.compile(self.pipe.unet, backend="cudagraphs")

        # self.pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=self.pipe.device, dtype=self.pipe.dtype)
        # self.pipe.vae = torch.compile(self.pipe.vae, backend="cudagraphs")

        self.latent_height = int(self.height // self.pipe.vae_scale_factor)
        self.latent_width = int(self.width // self.pipe.vae_scale_factor)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            logging.warning("Cannot use xformers memory efficient attention (maybe xformers not installed)")

        self.negative_prompt_embeds = None
        self.negative_prompt = ""
        if self.use_negative_prompt:
            self.negative_prompt = "lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck, illustration, painting, drawing, art, sketch,bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, worst quality, cropped, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, deformed, missing limb, bad hands, extra digits, extra fingers, not enough fingers, floating head, disembodied"
            negative_prompt_tokens = self.pipe.tokenizer(self.negative_prompt,
                                                         padding="max_length",
                                                         max_length=self.pipe.tokenizer.model_max_length,
                                                         truncation=True,
                                                         return_tensors="pt", ).to(self.pipe.text_encoder.device)
            self.negative_prompt_embed = self.pipe.text_encoder(negative_prompt_tokens.input_ids)[0]

    @torch.no_grad()
    def generate_image(self, embeddings: Tensor, latents: Tensor, loading_progress, queue_lock) -> list[Image]:
        """
        Generates a list of image(s) from given embedding

        Args:
        embedding (Tensor]):
            A single embedding as tensor of shape (batch, 77, 768)
        Returns:
            `list[PIL.Image.Image]: a list of batch many PIL images generated from the embeddings.
        """
        if embeddings.dtype != self.pipe.dtype:
            embeddings = embeddings.type(self.pipe.dtype)
        embeddings = embeddings.to(self.pipe.device)
        latents = latents.to(self.pipe.device)
        latents = latents.type(self.pipe.dtype)

        pos_prompt_embeds = embeddings
        num_embeddings = pos_prompt_embeds.shape[0]
        batch_steps = self.batch_size or num_embeddings

        images = []
        for i in range(0, num_embeddings, batch_steps):
            task = lambda: self.pipe(height=self.height,
                                     width=self.width,
                                     num_images_per_prompt=1,
                                     prompt_embeds=pos_prompt_embeds[i:i + batch_steps],
                                     negative_prompt_embeds=self.negative_prompt_embed.repeat(batch_steps, 1,
                                                                                              1) if self.use_negative_prompt else None,
                                     num_inference_steps=self.num_inference_steps,
                                     guidance_scale=self.guidance_scale,
                                     latents=latents[i:i + batch_steps],
                                     callback_on_step_end=partial(self.callback,
                                                                  current_step=i,
                                                                  num_embeddings=num_embeddings,
                                                                  loading_progress=loading_progress,
                                                                  batch_size=batch_steps,
                                                                  num_steps=self.num_inference_steps
                                                                  )
                                     ).images

            result = queue_lock.do_work(task)
            images.extend(result.result())
        self.latest_images.extend(images)
        return images



