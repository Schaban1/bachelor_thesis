import torch
from diffusers import StableDiffusionPipeline, StableDiffusionIPAdapterPipeline
from transformers import CLIPVisionModelWithProjection
from PIL import Image

class Generator:
    def __init__(self, pipe, num_inference_steps, guidance_scale, use_negative_prompt, batch_size, initial_latent_seed, device, cache_dir, hf_model_name):
        self.pipe = pipe or StableDiffusionPipeline.from_pretrained(hf_model_name, cache_dir=cache_dir)
        self.pipe.to(device)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_negative_prompt = use_negative_prompt
        self.batch_size = batch_size
        self.device = device
        self.latest_images = []
        self.callback = None
        generator = torch.Generator(device=device)
        generator.manual_seed(initial_latent_seed)
        self.generator = generator
        self.splice = None  # Set in app.py
        # Initialize IP-Adapter Plus
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                                                      torch_dtype=torch.float16).to(device)
        self.ip_pipe = StableDiffusionIPAdapterPipeline.from_pretrained(
            hf_model_name,
            image_encoder=image_encoder,
            torch_dtype=torch.float16
        ).to(device)
        self.ip_pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin"
        )
        self.ip_pipe.set_ip_adapter_scale(0.8)

    def generate_image(self, embeddings, latents, callback, queue_lock):
        images = []
        num_embeddings = embeddings.shape[0]
        current_step = 0
        for i in range(0, num_embeddings, self.batch_size):
            batch_size = min(self.batch_size, num_embeddings - i)
            batch_embeddings = embeddings[i:i + batch_size]
            batch_latents = embeddings[i:i + batch_size] if latents is not None else None
            with queue_lock:
                batch_images = self.pipe(
                    prompt_embeds=batch_embeddings,
                    latents=batch_latents,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    callback_on_step_end=lambda pipe, step_idx, timestep, cb_kwargs: self.callback(
                        pipe, step_idx, timestep, cb_kwargs, current_step, num_embeddings, callback, batch_size,
                        self.num_inference_steps
                    ) if self.callback else cb_kwargs,
                    generator=self.generator
                ).images
            images.extend(batch_images)
            current_step += batch_size
        self.latest_images = images
        return images

    def generate_with_splice(self, image: Image.Image, slider_values: dict, callback, queue_lock):
        # Preprocess image
        preprocessed = self.generator.vlm_backbone.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        # Decompose image to get SpLiCE weights
        sparse_weights = self.splice.encode_image(preprocessed)  # Shape: [1, vocab_size]
        # Adjust weights based on slider values
        for concept_idx, value in slider_values.items():
            sparse_weights[0, concept_idx] = max(0, min(1, sparse_weights[0, concept_idx] + value))  # Clamp to [0, 1]
        # Generate new embedding using SpLiCE
        new_embedding = self.splice.recompose_image(sparse_weights)  # Shape: [1, 1024]
        # Use IP-Adapter Plus for img2img
        with queue_lock:
            edited_image = self.ip_pipe(
                prompt="",
                image=image,  # img2img mode
                ip_adapter_image_embeds=[new_embedding],
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                callback_on_step_end=lambda pipe, step_idx, timestep, cb_kwargs: callback(
                    pipe, step_idx, timestep, cb_kwargs, 0, 1, callback, 1, self.num_inference_steps
                ) if callback else cb_kwargs,
                generator=self.generator
            ).images[0]
        return [edited_image]

    def get_latest_images(self):
        return self.latest_images

    def clear_latest_images(self):
        self.latest_images = []