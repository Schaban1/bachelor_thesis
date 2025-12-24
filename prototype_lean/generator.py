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

        self.splice = get_splice_model()

        SAE_PATH = RESOURCES_DIR / "sparse_autoencoder_final.pt"
        config = SparseAutoencoderConfig(
            n_input_features=1024,
            n_learned_features=8192,
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

        # MAIN PIPELINE: TEXT-TO-IMAGE
        self.pipe = pipe if pipe else StableDiffusionPipeline.from_pretrained(
            hf_model_name,
            requires_safety_checker=True,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
        ).to(device=self.device)

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdv1-5"
        )
        self.pipe.fuse_lora()

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

        # IP-ADAPTER PIPELINE
        self.ip_pipe = StableDiffusionPipeline.from_pretrained(
                hf_model_name,
                requires_safety_checker=True,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16,
            ).to(device=self.device)

        self.ip_pipe.scheduler = EulerDiscreteScheduler.from_config(self.ip_pipe.scheduler.config)
        #self.ip_pipe.scheduler = LCMScheduler.from_config(self.ip_pipe.scheduler.config)
        #self.ip_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        #self.ip_pipe.fuse_lora()

        """
        self.ip_pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                cache_dir=CACHE_DIR,
                weight_name="ip-adapter_sd15.bin",
            )
        self.ip_pipe.set_ip_adapter_scale(1.5)
        """
        try:
            self.ip_pipe.enable_xformers_memory_efficient_attention()
        except:
            logging.warning("Cannot use xformers in IP pipe")


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

    @torch.no_grad()
    def generate_with_splice(
            self,
            base_image: Image,
            concept_embedding: torch.Tensor,  # [1, 1024]
            loading_progress,
            queue_lock
    ) -> Image:
        self.initial_latent_generator.manual_seed(self.initial_latent_seed)

        concept_embedding = concept_embedding.unsqueeze(0)
        concept_embedding = concept_embedding.to(dtype=torch.float16, device=self.device)
        task = lambda: self.ip_pipe(
            height=self.height,
            width=self.width,
            num_images_per_prompt=1,
            prompt="",
            negative_prompt_embeds=(
                self.negative_prompt_embed.repeat(1, 1, 1)
                if self.use_negative_prompt else None
            ),
            num_inference_steps=50,
            guidance_scale=self.guidance_scale,
            strength=0.65,
            latents=None,
            generator=self.initial_latent_generator,
            callback_on_step_end=partial(
                self.callback,
                current_step=0,
                num_embeddings=1,
                loading_progress=loading_progress,
                batch_size=1,
                num_steps=self.num_inference_steps,
            ),
            #image=base_image,
            ip_adapter_image_embeds=[concept_embedding]
        ).images[0]

        result = queue_lock.do_work(task)
        return result.result()  # single PIL image
