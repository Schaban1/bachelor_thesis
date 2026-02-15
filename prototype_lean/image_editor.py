from collections import defaultdict
import torch
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import os
import hashlib
import splice

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

    def splice_edit(self, base_prompt: str, concept_offsets: dict, image_idx: int=0, loading_progress=None, queue_lock=None):
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

        eps = 1e-6

        # --- ORIGINAL NORM ---
        orig_norm = original_clip_feat.norm(dim=-1, keepdim=True)
        print(f"[DEBUG] original norm: {orig_norm.item():.6f}", flush=True)

        clip_feat_norm = original_clip_feat / orig_norm.clamp_min(eps)

        norm_after_norm = clip_feat_norm.norm(dim=-1)
        print(f"[DEBUG] norm after normalization (should be 1.0): {norm_after_norm.item():.6f}", flush=True)

        acts_original = self.sae.encode(clip_feat_norm)
        recon_original = self.sae.decode(acts_original)

        acts_modified = acts_original.clone()

        per_step_scale = 0.15

        for concept_idx, offset in concept_offsets.items():
            base_val = acts_modified[0, concept_idx].item()
            steps = int(abs(offset) / 0.1)
            direction = 1.0 if offset > 0 else -1.0

            delta = direction * steps * per_step_scale * base_val
            acts_modified[0, concept_idx] = torch.clamp(
                acts_modified[0, concept_idx] + delta,
                min=0.0
            )

        recon_modified = self.sae.decode(acts_modified)

        steering_delta = recon_modified - recon_original
        target_feat_norm = clip_feat_norm + steering_delta

        # --- BEFORE RENORMALIZATION ---
        pre_norm = target_feat_norm.norm(dim=-1)
        print(f"[DEBUG] target norm before renorm: {pre_norm.item():.6f}", flush=True)

        # re-scale back to original magnitude
        target_feat = target_feat_norm / target_feat_norm.norm(dim=-1, keepdim=True).clamp_min(eps)
        target_feat = target_feat * orig_norm

        # --- AFTER RENORMALIZATION ---
        final_norm = target_feat.norm(dim=-1)
        print(f"[DEBUG] final norm after rescale: {final_norm.item():.6f}", flush=True)

        new_img = self.generator.generate_with_sae(
            base_image, target_feat, loading_progress, queue_lock
        )

        img_hash = hashlib.md5(new_img.tobytes()).hexdigest()[:8]
        print(f"[GENERATOR] New Image {image_idx} Created | Pixel Hash: {img_hash}", flush=True)

        self.cache[image_idx][state_key] = new_img
        return new_img
