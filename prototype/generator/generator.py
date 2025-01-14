from abc import abstractmethod, ABC
from torch import Tensor

from nicegui import binding
import torch
from PIL.Image import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL, AutoencoderTiny
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
import time

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
                 batch_size: int = None,
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

        self.device = torch.device("cuda") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=cache_dir,
            torch_dtype=torch.float16
        ).to(device=self.device)

        self.stream = StreamDiffusion(
            self.pipe,
            t_index_list=[0, 16, 32, 45],
            torch_dtype=torch.float16,
            cfg_type="none",
        )
        self.stream.load_lcm_lora()
        self.stream.fuse_lora()

        self.stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.pipe.dtype).to(device=self.pipe.device)
        #stream.vae = AutoencoderKL.from_pretrained("").to(device=pipe.device, dtype=pipe.dtype)

        self.latent_height = int(self.height // self.pipe.vae_scale_factor)
        self.latent_width = int(self.width // self.pipe.vae_scale_factor)

        self.pipe.enable_xformers_memory_efficient_attention()
        # if torch.onnx.is_onnxrt_backend_supported():
        #     print("compiling...")
        #     self.pipe.unet = torch.compile(self.pipe.unet, backend="onnxrt")

        self.load_generator()

    def load_generator(self):
        self.latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.height, self.width),
            device=self.pipe.device, dtype=self.pipe.dtype
        ).repeat(n_images, 1, 1, 1)

        self.negative_prompt_embeds = None
        self.negative_prompt = ""
        if self.use_negative_prompt:
            self.negative_prompt = "lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck, illustration, painting, drawing, art, sketch,bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, worst quality, cropped, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, deformed, missing limb, bad hands, extra digits, extra fingers, not enough fingers, floating head, disembodied"
            negative_prompt_tokens = self.pipe.tokenizer(self.negative_prompt,
                                                         padding="max_length",
                                                         max_length=self.pipe.tokenizer.model_max_length,
                                                         truncation=True,
                                                         return_tensors="pt", ).to(self.pipe.text_encoder.device)
            self.negative_prompt_embeds = self.pipe.text_encoder(negative_prompt_tokens.input_ids)[0].repeat(self.n_images, 1, 1)

    def setup(self, prompt: str, seed: int):
        self.stream.prepare(prompt=prompt,
                            negative_prompt=self.negative_prompt,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            seed=seed)
        for _ in range(4):
            self.stream()


    @torch.no_grad()
    def generate_image_stream(self, embeddings: Tensor | tuple[Tensor, Tensor], latents: Tensor = None) -> list[Image]:
        start = time.time()
        latents = latents.to(self.pipe.device)
        embeddings = embeddings.to(self.pipe.device)
        embeddings = embeddings.type(self.pipe.dtype)

        if latents != None:
            latents = latents.to(self.pipe.device)
            latents = latents.type(self.pipe.dtype)
        else:
            if self.random_latents:
                latents = torch.randn(
                    (self.n_images, self.pipe.unet.config.in_channels, self.latent_height, self.latent_width),
                    device=self.pipe.device, dtype=self.pipe.dtype
                )
            else:
                latents = self.latents

        image_list = []
        for embedding, latent in zip(embeddings, latents):
            embedding = embedding.repeat(self.stream.batch_size, 1, 1)
            latent = latent.unsqueeze(0)

            self.stream.prompt_embeds = embedding
            x_0_pred_out = self.stream.predict_x0_batch(latent)

            x_output = self.stream.decode_image(x_0_pred_out).detach().clone()
            image_list += postprocess_image(x_output, output_type="pil")
        print(f"generation done in {time.time() - start}")
        return image_list

    @torch.no_grad()
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
            latents = latents.to(self.pipe.device)
            latents = latents.type(self.pipe.dtype)
        else:
            if self.random_latents:
                latents = torch.randn(
                    (self.n_images, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
                    device=self.pipe.device, dtype=self.pipe.dtype
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
        return images


if __name__ == "__main__":
    import os
    import time

    n_images = 5
    gen = Generator(n_images=n_images,
                    batch_size=None,
                    cache_dir=None,
                    num_inference_steps=50,
                    use_negative_prompt=False,
                    random_latents=True)

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    # prompt_tokens = gen.pipe.tokenizer(prompt,
    #                                    padding="max_length",
    #                                    max_length=gen.pipe.tokenizer.model_max_length,
    #                                    truncation=True,
    #                                    return_tensors="pt",
    #                                    ).to(gen.device)

    embed = gen.pipe.encode_prompt(prompt,
                                   device=gen.pipe.device,
                                   num_images_per_prompt=1,
                                   do_classifier_free_guidance=False)[0]
    embed = embed.repeat(n_images, 1, 1)
    print(f"{embed.shape=}")

    start = time.time()
    os.makedirs("output", exist_ok=True)
    for i in range(1):
        img = gen.generate_image(embed)
        for i in range(n_images):
            img[i].save(f"output/{i}.png")
    print("normal generation took:", time.time() - start)

    print("running stream")
    start = time.time()
    os.makedirs("output_stream", exist_ok=True)
    for i in range(1):
        img = gen.generate_image_stream(embed)
        for i in range(n_images):

            img[i].save(f"output_stream/{i}.png")
    print("stream generation took:", time.time() - start)
    print(img)
