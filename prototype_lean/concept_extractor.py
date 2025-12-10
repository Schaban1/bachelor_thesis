import torch
import splice
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import os

class SpliceExtractor:
    def __init__(self, splice_model):
        self.splice = splice_model
        self.vocabulary = splice.get_vocabulary("mscoco", 10000)

    def extract_top_concepts(self, pil_image, topk=5):

        sparse_weights = self.splice.encode_image(pil_image)
        print(f"[DEBUG] sparse_weights → type: {type(sparse_weights)}", flush=True)
        print(f"[DEBUG] sparse_weights → shape: {sparse_weights.shape}", flush=True)

        topk_indices = torch.topk(sparse_weights[0], k=topk).indices.tolist()
        topk_values = sparse_weights[0, topk_indices].tolist()

        concepts = [self.vocabulary[i] for i in topk_indices]

        print(f"[DEBUG SPLICE] concepts → type: {type(concepts)} | length: {len(concepts)}", flush=True)
        print(f"[DEBUG SPLICE] concepts → {concepts}", flush=True)
        print(f"[DEBUG SPLICE] values → {topk_values}", flush=True)
        print(f"[DEBUG SPLICE] final return → {list(zip(concepts, topk_indices))}", flush=True)
        print("[DEBUG SPLICE conceptsextractor: were the concepts extracted?]?", flush=True)
        return list(zip(concepts, topk_values, topk_indices))


class SAEExtractor:
    def __init__(self, sae, concept_names):
        self.device = "cuda"
        CACHE_DIR = str(Path(__file__).resolve().parent / "cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[CACHE] SAEExtractor using cache: {CACHE_DIR}")
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=CACHE_DIR).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",cache_dir=CACHE_DIR)

        self.sae = sae
        self.concept_names = concept_names

    @torch.no_grad()
    def extract_top_concepts(self, pil_image, top_k=5):
        inputs = self.clip_processor(images=pil_image,return_tensors="pt")["pixel_values"].to(self.device)
        clip_feat = self.clip_model.get_image_features(inputs)

        acts = self.sae.encode(clip_feat)               # → (1,8192)
        acts = acts.squeeze(0).cpu().numpy()

        top_idx = acts.argsort()[-top_k:][::-1]
        top_values = acts[top_idx]
        concepts = [self.concept_names[i] for i in top_idx]
        print(f"[DEBUG SAE] concepts → type: {type(concepts)} | length: {len(concepts)}", flush=True)
        print(f"[DEBUG SAE] concepts → {concepts}", flush=True)
        print(f"[DEBUG SAE] values → {top_values.tolist()}", flush=True)
        print(f"[DEBUG SAE] final return → {list(zip(concepts, top_values.tolist(), top_idx.tolist()))}", flush=True)
        print("[DEBUG SAE: were the concepts extracted?]", flush=True)
        return list(zip(concepts, top_values.tolist(), top_idx.tolist()))