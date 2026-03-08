from collections import defaultdict
import torch
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import os
import hashlib
import splice
from constants import RESOURCES_DIR

class ImageEditor:
    def __init__(self, generator):
        self.generator = generator
        self.splice = generator.splice  # pre-loaded
        self.sae = generator.sae_model

        self.vocabulary = splice.get_vocabulary("laion", 10000)

        CACHE_DIR = str(Path(__file__).resolve().parent / "cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[CACHE] ImageEditor SAE using cache: {CACHE_DIR}")

        self.device = "cuda"
        self.clip_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            cache_dir=CACHE_DIR
        ).to(self.device).eval()

        self.clip_processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            cache_dir=CACHE_DIR
        )

        # Caching: (image_idx, base_prompt_hash, state_key) -> PIL image
        self.cache = defaultdict(dict)


        self.sae_vocab_index = {}
        csv_path = RESOURCES_DIR / "concept_names.csv"
        with open(csv_path, "r") as f:
            for line in f:
                idx, name = line.strip().split(",")
                self.sae_vocab_index[name] = int(idx)

    def splice_edit(self, base_prompt: str, concept_offsets: dict, image_idx: int = 0, loading_progress=None,
                    queue_lock=None):
        # Deterministic cache key
        base_hash = hashlib.md5(base_prompt.encode()).hexdigest()[:8]
        state_items = sorted(concept_offsets.items(), key=lambda x: str(x[0]))
        state_key = tuple(state_items)

        cache_key = (int(image_idx), base_hash, state_key)
        if cache_key in self.cache:
            print(
                f"[CACHE HIT] Returning cached image for image_idx={image_idx} "
                f"prompt='{base_prompt}' | offsets {state_key}"
            )
            return self.cache[cache_key]

        text_inputs = self.generator.pipe.tokenizer(
            base_prompt,
            padding="max_length",
            max_length=self.generator.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.generator.device)

        with torch.no_grad():
            text_embeds = self.generator.pipe.text_encoder(text_inputs.input_ids)[0]

        base_emb = text_embeds[:, -1:, :].squeeze(1)  # summary token

        # Decomposition
        weights = self.splice.encode_image(base_emb)

        for concept, offset in concept_offsets.items():
            if isinstance(concept, str):
                idx = self.vocabulary.index(concept)
            else:
                idx = concept

            base_val = weights[0, idx].item()

            steps = int(abs(offset) / 0.1)
            direction = 1.0 if offset > 0 else -1.0

            delta = direction * steps * 0.25
            new_val = base_val + delta

            weights[0, idx] = torch.clamp(
                torch.tensor(new_val, device=weights.device),
                min=0.0
            )

        # Recompose
        recon = self.splice.recompose_image(weights)

        # Denormalize
        denormalized = recon / torch.std(recon) * torch.std(base_emb)
        denormalized = denormalized - torch.mean(denormalized) + torch.mean(base_emb)

        print("[DEBUG] denormalized device, dtype, norm, nan:", denormalized.device, denormalized.dtype,
              float(denormalized.norm()), torch.isnan(denormalized).any(), flush=True)

        # Expand to full prompt_embeds
        original_starttoken = text_embeds[:, 0, :].detach()

        def convert_to_full_text(summary_vec, original_starttoken, n_tokens=77):
            if summary_vec.dim() == 1:
                summary_vec = summary_vec.unsqueeze(0)
            B, D = summary_vec.shape
            first = original_starttoken.reshape(1, 1, D).expand(B, 1, D)
            rest = summary_vec.reshape(B, 1, D).expand(B, n_tokens - 1, D)
            full = torch.cat([first, rest], dim=1)
            return full

        summary_vec = denormalized.view(denormalized.shape[0], -1)
        prompt_emb_full = convert_to_full_text(summary_vec, original_starttoken, n_tokens=77)

        prompt_emb_full = prompt_emb_full.to(
            device=self.generator.edit_pipe.device,
            dtype=self.generator.edit_pipe.unet.dtype
        )

        print("[DEBUG] prompt_emb_full device/dtype/min/max:", prompt_emb_full.device, prompt_emb_full.dtype,
              float(prompt_emb_full.min()), float(prompt_emb_full.max()), flush=True)

        # Generate
        images = self.generator.generate_with_splice(prompt_emb_full, loading_progress,
                                                     queue_lock=queue_lock, image_idx=image_idx)

        result_img = images[0]

        # Cache
        self.cache[cache_key] = result_img
        print(f"[CACHE] Saved new image for prompt '{base_prompt}' | offsets {state_key}")

        return result_img

    @torch.no_grad()
    def sae_edit(self, base_prompt: str, concept_offsets: dict, image_idx: int = 0, loading_progress=None, queue_lock=None):
        base_hash = hashlib.md5(base_prompt.encode()).hexdigest()[:8]
        state_items = sorted(concept_offsets.items(), key=lambda x: str(x[0]))
        state_key = tuple(state_items)
        cache_key = (int(image_idx), base_hash, state_key)

        if cache_key in self.cache:
            print(f"[CACHE HIT] sae_edit for image {image_idx}", flush=True)
            return self.cache[cache_key]

        text_inputs = self.generator.pipe.tokenizer(
            base_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(self.generator.device)

        with torch.no_grad():
            text_embeds = self.generator.pipe.text_encoder(text_inputs.input_ids)[0]

        total_direction = torch.zeros(
            (1, text_embeds.shape[-1]),
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )
        for concept_idx, offset in concept_offsets.items():
            if offset == 0:
                continue
            concept_name = self.vocabulary[concept_idx] if isinstance(concept_idx, int) else str(concept_idx)
            direction = self.build_direction(concept_name)
            if direction.dim() == 1:
                direction = direction.unsqueeze(0)
            total_direction = total_direction + float(offset) * direction

        prompt_emb_full = self.edit_with_direction(
            base_prompt,
            total_direction,
            strength=1.0,
        )

        target_device = getattr(self.generator.edit_pipe, "device", self.generator.device)
        target_dtype = getattr(self.generator.edit_pipe.unet, "dtype", torch.float32)
        prompt_emb_full = prompt_emb_full.to(device=target_device, dtype=target_dtype)

        sae_image_idx = image_idx + 100

        images = self.generator.generate_with_splice(
            prompt_emb_full,
            loading_progress,
            queue_lock,
            image_idx=sae_image_idx,
            is_sae=True
        )

        result_img = images[0]

        self.cache[cache_key] = result_img
        print(f"[SAE EDIT] New image for prompt '{base_prompt}' | offsets {state_key}", flush=True)
        return result_img
    def build_direction(self, concept_name):
        pos_templates = [
            "a {}", "the {}", "this is a {}", "image of a {}", "photo of a {}",
            "picture of {}", "close-up of a {}", "detailed {}", "beautiful {}",
            "realistic {}", "{}", "{} object", "{} scene", "{} in nature"
        ]
        neg_templates = ["", "nothing", "empty scene", "blank", "no {}", "without {}"]
        pos_prompts = [t.format(concept_name) for t in pos_templates * 2]
        neg_prompts = [t.format(concept_name) for t in neg_templates * 2]

        pos_emb = torch.mean(torch.stack([
            self.generator.pipe.text_encoder(self.generator.pipe.tokenizer(p, return_tensors="pt").to(self.device).input_ids)[0][:, -1, :]
            for p in pos_prompts
        ]), dim=0)
        neg_emb = torch.mean(torch.stack([
            self.generator.pipe.text_encoder(self.generator.pipe.tokenizer(p, return_tensors="pt").to(self.device).input_ids)[0][:, -1, :]
            for p in neg_prompts
        ]), dim=0)

        direction = (pos_emb - neg_emb)
        direction = direction / (direction.norm() + 1e-8)
        return direction

    def edit_with_direction(self, original_prompt, direction, strength=1.0):
        inputs = self.generator.pipe.tokenizer(
            original_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            text_embeds = self.generator.pipe.text_encoder(inputs.input_ids)[0]
        summary = text_embeds[:, -1, :]
        edited_summary = summary + strength * direction
        full = torch.cat([
            text_embeds[:, 0, :].unsqueeze(1),
            edited_summary.unsqueeze(1).expand(-1, 76, -1)
        ], dim=1)
        return full

    def _convert_to_full_text(self, summary_vec, original_starttoken, n_tokens=77):
        if summary_vec.dim() == 1:
            summary_vec = summary_vec.unsqueeze(0)
        if original_starttoken.dim() == 1:
            original_starttoken = original_starttoken.unsqueeze(0)
        k = summary_vec.shape[-1]
        return torch.cat([
            original_starttoken.reshape([1, 1, k]).expand(summary_vec.shape[0], 1, -1),
            summary_vec.reshape([1, 1, k]).expand(summary_vec.shape[0], n_tokens - 1, -1)
        ], dim=1)