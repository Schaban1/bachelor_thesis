import logging
import torch
from PIL.Image import Image
from abc import abstractmethod, ABC
from diffusers import StableDiffusionPipeline, LCMScheduler, EulerDiscreteScheduler
from functools import partial
from nicegui import binding
from torch import Tensor
from nicegui import ui as ngUI
from pathlib import Path
import os
from constants import RESOURCES_DIR

import torch.optim.optimizer as opt_fix
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Iterable, Dict, Any
from unittest.mock import MagicMock
import sys

if not hasattr(opt_fix, "params_t"):
    opt_fix.params_t = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class MockHookPoint(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

tl_module = MagicMock()
tl_module.HookedTransformer = MagicMock

hp_module = MagicMock()
hp_module.HookPoint = MockHookPoint
utils_module = MagicMock()

sys.modules["transformer_lens"] = tl_module
sys.modules["transformer_lens.hook_points"] = hp_module
sys.modules["transformer_lens.utils"] = utils_module

tl_module.hook_points = hp_module
tl_module.utils = utils_module


def get_bias_tensor(module_or_tensor):
    """
    Extracts the raw tensor from a bias module or returns the tensor itself
    """
    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor

    if hasattr(module_or_tensor, '_bias_reference'):
        return module_or_tensor._bias_reference

    if hasattr(module_or_tensor, 'bias'):
        return module_or_tensor.bias

    return module_or_tensor


def manual_encode_fix(self, x):
    # Logic: (Input - PreBias) -> Encoder -> Activation

    #x = x / x.norm(dim=-1, keepdim=True)

    # 1. Apply pre-encoder bias
    if hasattr(self, 'pre_encoder_bias'):
        bias = get_bias_tensor(self.pre_encoder_bias)
        x = x - bias

    # 2. Linear Encoding Layer
    x = self.encoder(x)

    # 3. Activation
    if hasattr(self, 'activation'):
        x = self.activation(x)
    else:
        x = F.relu(x)
    return x


def manual_decode_fix(self, f):
    # Logic: Features -> Decoder -> + PostBias (Reconstruction)
    x = self.decoder(f)

    # 1. Apply post-decoder bias
    if hasattr(self, 'post_decoder_bias'):
        bias = get_bias_tensor(self.post_decoder_bias)
        x = x + bias

    return x

from sparse_autoencoder import SparseAutoencoder,SparseAutoencoderConfig
SparseAutoencoder.encode = manual_encode_fix
SparseAutoencoder.decode = manual_decode_fix
from splice_custom import get_splice_model

class GeneratorBase(ABC):
    def __init__(self):
        self.latest_images = []

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass

    def get_latest_images(self) -> list[Image]:
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
                 ):
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

        SAE_PATH = RESOURCES_DIR / "sparse_autoencoder_final.pt"
        config = SparseAutoencoderConfig(
            n_input_features=768,
            n_learned_features=6144,
        )

        self.sae_model = SparseAutoencoder(config).to(self.device)
        print(f"Loading SAE from: {SAE_PATH}")
        state_dict = torch.load(SAE_PATH, map_location=self.device)

        print("Fixing dimensions in state_dict...")
        for key in list(state_dict.keys()):
            tensor = state_dict[key]
            if len(tensor.shape) > 0 and tensor.shape[0] == 1:
                state_dict[key] = tensor.squeeze(0)
                print(f"Fixed {key}: {tensor.shape} -> {state_dict[key].shape}")

        self.sae_model.load_state_dict(state_dict, strict=False)
        self.sae_model.eval()

        os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / "cache")
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
        os.environ["DIFFUSERS_CACHE"] = os.environ["HF_HOME"]
        os.environ["TORCH_HOME"] = os.environ["HF_HOME"]

        CACHE_DIR = os.environ["HF_HOME"]
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[CACHE] ALL MODELS → {CACHE_DIR}")

        os.environ["TORCH_SDPA_DISABLE_FLASH_ATTENTION"] = "1"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # MAIN PIPELINE: TEXT-TO-IMAGE
        self.pipe = pipe if pipe else StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR,
        ).to(self.device)

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdv1-5"
        )
        self.pipe.fuse_lora()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            logging.warning("Cannot use xformers memory efficient attention (maybe xformers not installed)")

        uncond_input = self.pipe.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt").to(
            self.device)
        with torch.no_grad():
            self.uncond_embeds = self.pipe.text_encoder(uncond_input.input_ids)[0]

        self.initial_latent_generator = torch.Generator(device=self.device).manual_seed(42)
        self.latents_fixed = torch.randn((1, self.pipe.unet.in_channels, self.height // 8, self.width // 8),
                                         generator=self.initial_latent_generator, device=self.device,
                                         dtype=self.pipe.dtype)

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

        # IP-ADAPTER PIPELINE
        self.ip_pipe = StableDiffusionPipeline.from_pretrained(
                hf_model_name,
                requires_safety_checker=True,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16,
            ).to(device=self.device)

        self.ip_pipe.scheduler = LCMScheduler.from_config(self.ip_pipe.scheduler.config)
        self.ip_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.ip_pipe.fuse_lora()

        try:
            self.ip_pipe.enable_xformers_memory_efficient_attention()
        except:
            logging.warning("Cannot use xformers in IP pipe")

        self.splice = get_splice_model(self.pipe, self.device)

        # --- EDIT PIPE ---
        self.edit_pipe = StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR,
        ).to(self.device)

        self.edit_pipe.scheduler = LCMScheduler.from_config(self.edit_pipe.scheduler.config)
        try:
            self.edit_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self.edit_pipe.fuse_lora()
        except Exception:
            print("[WARN] edit_pipe: LORA not available or failed", flush=True)


        uncond_input_edit = self.edit_pipe.tokenizer(
            [""], padding="max_length", max_length=77, return_tensors="pt"
        )

        uncond_input_edit = {k: v.to(self.device) for k, v in uncond_input_edit.items()}
        with torch.no_grad():
            self.edit_uncond_embeds = self.edit_pipe.text_encoder(uncond_input_edit["input_ids"])[0]

        self.edit_latent_generator = torch.Generator(device=self.device).manual_seed(self.initial_latent_seed)
        self.edit_latents_fixed = torch.randn(
            (1, self.edit_pipe.unet.in_channels, self.height // 8, self.width // 8),
            generator=self.edit_latent_generator,
            device=self.device,
            dtype=torch.float32
        )

        print("[INIT] edit_pipe ready: dtype", self.edit_pipe.unet.dtype, "latents dtype", self.edit_latents_fixed.dtype, flush=True)




    @torch.no_grad()
    def generate_image(self, embeddings: Tensor, latents: Tensor, loading_progress, queue_lock) -> list[Image]:
        """
        Generates a list of image(s) from given embedding
        """
        self.initial_latent_generator.manual_seed(self.initial_latent_seed)
        if embeddings.dtype != self.pipe.dtype:
            embeddings = embeddings.type(self.pipe.dtype)
        embeddings = embeddings.to(self.pipe.device)
        #latents = latents.to(self.pipe.device)
        #latents = latents.type(self.pipe.dtype)

        pos_prompt_embeds = embeddings
        num_embeddings = pos_prompt_embeds.shape[0]
        batch_steps = self.batch_size or num_embeddings

        images = []
        for i in range(0, num_embeddings, batch_steps):
            task = lambda: self.pipe(height=self.height,
                                     width=self.width,
                                     num_images_per_prompt=1,
                                     prompt_embeds=pos_prompt_embeds[i:i + batch_steps],
                                     negative_prompt_embeds=self.negative_prompt_embed.repeat(batch_steps, 1, 1) if self.use_negative_prompt else None,
                                     num_inference_steps=self.num_inference_steps,
                                     guidance_scale=self.guidance_scale,
                                     latents=None,
                                     generator=self.initial_latent_generator,
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

        print(f"[GENERATOR] FINAL: returning {len(images)} images",flush=True)
        self.latest_images.extend(images)

        return images

    def _run_manual_loop(self, prompt_embeds, num_inference_steps: int, guidance_scale: float):

        step_generator = torch.Generator(device=self.pipe.device).manual_seed(self.initial_latent_seed)
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.pipe.device)
        latents_curr = self.latents_fixed.clone().to(self.pipe.device, dtype=self.pipe.unet.dtype)

        prompt_embeds = prompt_embeds.to(device=self.pipe.device, dtype=self.pipe.unet.dtype)
        uncond_embeds = self.uncond_embeds.to(device=self.pipe.device, dtype=self.pipe.unet.dtype)

        print("[DEBUG] _run_manual_loop: latents_curr.device/dtype:", latents_curr.device, latents_curr.dtype, flush=True)
        print("[DEBUG] _run_manual_loop: prompt_embeds.device/dtype:", prompt_embeds.device, prompt_embeds.dtype, flush=True)
        print("[DEBUG] _run_manual_loop: uncond_embeds.device/dtype:", uncond_embeds.device, uncond_embeds.dtype, flush=True)
        print("[DEBUG] _run_manual_loop: scheduler timesteps len:", len(self.pipe.scheduler.timesteps), flush=True)

        for t in self.pipe.scheduler.timesteps:
            latent_in = torch.cat([latents_curr] * 2).to(self.pipe.unet.dtype)
            latent_in = self.pipe.scheduler.scale_model_input(latent_in, t)
            model_in_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)

            with torch.no_grad():
                noise_pred = self.pipe.unet(latent_in, t, encoder_hidden_states=model_in_embeds).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            step_output = self.pipe.scheduler.step(noise_pred, t, latents_curr, generator=step_generator)
            latents_curr = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output[0]

        latents_curr = latents_curr.to(self.pipe.vae.dtype)
        decoded = self.pipe.vae.decode(latents_curr / self.pipe.vae.config.scaling_factor).sample
        image = (decoded / 2 + 0.5).clamp(0, 1).detach()
        return image


    @torch.no_grad()
    def generate_with_splice(self, prompt_embeds: torch.Tensor, loading_progress=None, queue_lock=None):
        num_inference_steps = 6
        guidance_scale = 1.0

        print("[DEBUG] generate_with_splice: incoming prompt_embeds shape/dtype/device:",
              getattr(prompt_embeds, "shape", None), getattr(prompt_embeds, "dtype", None), getattr(prompt_embeds, "device", None),
              flush=True)

        task = lambda: self._run_manual_loop(prompt_embeds, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        result = queue_lock.do_work(task) if queue_lock else task()
        images_tensor = result.result() if hasattr(result, "result") else result

        images = []
        for i in range(images_tensor.shape[0]):
            pil = self.pipe.image_processor.postprocess(images_tensor[i:i + 1], output_type='pil')[0]
            images.append(pil)

        self.latest_images.extend(images)
        return images

    @staticmethod
    def expand_to_prompt_embeds(x: torch.Tensor, seq_len: int = 77):
        """
        x: [768] oder [1,768]
        return: [1,77,768]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1,768]
        x = x.unsqueeze(1)  # [1,1,768]
        x = x.repeat(1, seq_len, 1)  # [1,77,768]
        return x

    @torch.no_grad()
    def generate_with_sae(
            self,
            base_image: Image,
            concept_embedding: torch.Tensor,
            loading_progress,
            queue_lock
    ) -> Image:
        self.initial_latent_generator.manual_seed(self.initial_latent_seed)

        concept_embedding = self.expand_to_prompt_embeds(concept_embedding)
        concept_embedding = concept_embedding.to(dtype=torch.float16, device=self.device)

        strength = 0.8
        num_inference_steps = 8
        guidance_scale = 8.0

        # PIL -> model tensor
        image = self.ip_pipe.image_processor.preprocess(
            base_image,
            height=self.height,
            width=self.width
        ).to(self.device, self.ip_pipe.dtype)

        latents = self.ip_pipe.vae.encode(image).latent_dist.sample()
        latents = latents * self.ip_pipe.vae.config.scaling_factor

        latents = latents.to(self.ip_pipe.device, dtype=self.ip_pipe.dtype)

        self.ip_pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        start_idx = int((1.0 - strength) * num_inference_steps)
        start_idx = min(start_idx, num_inference_steps - 1)
        timestep = self.ip_pipe.scheduler.timesteps[start_idx]

        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=self.initial_latent_generator,
        )

        latents = self.ip_pipe.scheduler.add_noise(latents, noise, timestep)

        task = lambda: self.ip_pipe(
            height=self.height,
            width=self.width,
            num_images_per_prompt=1,
            prompt_embeds=concept_embedding,
            negative_prompt_embeds=self.negative_prompt_embed.repeat(1, 1, 1) if self.use_negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.initial_latent_generator,
            latents= latents,
            callback_on_step_end=partial(
                self.callback,
                current_step=0,
                num_embeddings=1,
                loading_progress=loading_progress,
                batch_size=1,
                num_steps=self.num_inference_steps,
            ),

        ).images[0]

        result = queue_lock.do_work(task)
        return result.result()  # single PIL image
