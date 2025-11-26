# generator/image_editor.py
from collections import defaultdict
import torch
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import os

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
        state_items = sorted(concept_offsets.items())
        state_key = tuple(state_items)

        if state_key in self.cache[image_idx]:
            return self.cache[image_idx][state_key]

        inputs = self.clip_processor(images=base_image, return_tensors="pt")["pixel_values"].to(self.device)
        clip_feat = self.clip_model.get_image_features(inputs)

        steered = clip_feat.clone()
        decoder_normed = self.sae.decoder_weight / self.sae.decoder_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)

        for concept_idx, offset in concept_offsets.items():
            steered += offset * decoder_normed[:, concept_idx]

        steered = steered / steered.norm(dim=-1, keepdim=True)

        # Generate
        new_img = self.generator.generate_with_splice(
            base_image, steered, loading_progress, queue_lock
        )

        # Cache result
        self.cache[image_idx][state_key] = new_img
        return new_img