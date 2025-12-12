from collections import defaultdict
import torch
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import os
import hashlib

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
            cached_img = self.cache[image_idx][state_key]
            img_hash = hashlib.md5(cached_img.tobytes()).hexdigest()[:8]
            print(f"[CACHE HIT] Returning image {image_idx} | ID: {img_hash}", flush=True)
            return cached_img

        inputs = self.clip_processor(images=base_image, return_tensors="pt")["pixel_values"].to(self.device)

        original_clip_feat = self.clip_model.get_image_features(inputs)

        # normalized version for calculations
        #clip_feat_norm = original_clip_feat / original_clip_feat.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            acts_original = self.sae.encode(original_clip_feat)
            recon_original = self.sae.decode(acts_original)

            acts_modified = acts_original.clone()
            for concept_idx, offset in concept_offsets.items():
                new_val = acts_modified[0, concept_idx] + offset
                acts_modified[0, concept_idx] = torch.clamp(new_val, min=0.0)

            recon_modified = self.sae.decode(acts_modified)
            steering_delta = recon_modified - recon_original
            target_feat = original_clip_feat + steering_delta
            target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)

        # Generate using the new target embedding
        new_img = self.generator.generate_with_splice(
            base_image, target_feat, loading_progress, queue_lock
        )

        img_hash = hashlib.md5(new_img.tobytes()).hexdigest()[:8]
        print(f"[GENERATOR] New Image {image_idx} Created | Pixel Hash: {img_hash}", flush=True)

        # Cache result
        self.cache[image_idx][state_key] = new_img
        return new_img