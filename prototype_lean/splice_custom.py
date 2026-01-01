import torch
from transformers import CLIPModel, CLIPProcessor
import splice
from pathlib import Path
import os
from constants import RESOURCES_DIR

class VLMBackbone(torch.nn.Module):
    def __init__(self):
        super(VLMBackbone, self).__init__()
        CACHE_DIR = str(Path(__file__).resolve().parent / "cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[CACHE] splice_custom LOCKED TO: {CACHE_DIR}")
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", cache_dir=CACHE_DIR)
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", cache_dir=CACHE_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs)

    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs)

def get_splice_model(image_mean_path = RESOURCES_DIR / "image_mean_flickr_2k_vit_l14.pt", device="cuda"):
    # Setup backbone
    vlm_backbone = VLMBackbone()

    # Build vocabulary
    concepts = splice.get_vocabulary("laion", 10000)
    embedded_concepts = []
    for concept in concepts:
        emb = vlm_backbone.encode_text(concept)
        embedded_concepts.append(emb.squeeze(0))

    concepts_tensor = torch.stack(embedded_concepts)
    concepts_tensor = torch.nn.functional.normalize(concepts_tensor, dim=1)
    concepts_tensor = concepts_tensor - torch.mean(concepts_tensor, dim=0)
    concepts_tensor = torch.nn.functional.normalize(concepts_tensor, dim=1)

    # Load mean
    image_mean = torch.load(image_mean_path).to(device)

    # Return ready-to-use model
    splicemodel = splice.SPLICE(image_mean, concepts_tensor, clip=vlm_backbone, device=device, return_weights=True)
    return splicemodel