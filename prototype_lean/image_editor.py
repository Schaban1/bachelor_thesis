from collections import defaultdict
import torch
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import os
import torch.nn.functional as F

class ImageEditor:
    def __init__(self, generator, splice_model, sae_model):
        self.generator = generator
        self.splice = splice_model
        self.sae = sae_model

        CACHE_DIR = str(Path(__file__).resolve().parent / "cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[CACHE] ImageEditor SAE using cache: {CACHE_DIR}")

        self.device = "cuda"
        self.clip_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=CACHE_DIR
        ).to(self.device).eval()

        self.clip_processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=CACHE_DIR
        )

        self.cache = defaultdict(dict)  # (img_idx, state_key) -> PIL image

    def splice_edit(self, base_image, concept_offsets, image_idx, loading_progress=None, queue_lock=None):
        # Create deterministic cache key from (concept_idx, offset) pairs
        state_items = sorted(concept_offsets.items())
        state_key = tuple(state_items)

        if state_key in self.cache[image_idx]:
            return self.cache[image_idx][state_key]

        # Modify weights
        weights = self.splice.encode_image(base_image)
        for concept_idx, offset in concept_offsets.items():
            weights[0, concept_idx] = max(0, min(1, weights[0, concept_idx] + offset))

        # Recompose
        embedding = self.splice.recompose_image(weights)

        # Generate
        new_img = self.generator.generate_with_splice(
            base_image, embedding, loading_progress, queue_lock
        )

        # Cache
        self.cache[image_idx][state_key] = new_img
        return new_img

    @torch.no_grad()
    def sae_edit(self, base_image, concept_offsets, image_idx, loading_progress=None, queue_lock=None):
        if not concept_offsets:
            return base_image

        state_key = tuple(sorted(concept_offsets.items()))
        if state_key in self.cache[image_idx]:
            return self.cache[image_idx][state_key]

        # 1. Raw CLIP feature
        inputs = self.clip_processor(images=base_image, return_tensors="pt")["pixel_values"].to(self.device)
        clip_feat = self.clip_model.get_image_features(inputs)  # (1, 1024), norm ≈20

        # 2. SAE encode
        acts = self.sae.encode(clip_feat)  # (1, 8192)
        acts = acts.clone()

        # 3. Steering
        for idx, offset in concept_offsets.items():
            acts[0, idx] = F.relu(acts[0, idx] + offset)

        # 4. Decode
        steered = self.sae.decode(acts)
        steered = steered / steered.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 5. Generate
        new_img = self.generator.generate_with_splice(base_image, steered, loading_progress, queue_lock)

        self.cache[image_idx][state_key] = new_img
        return new_img