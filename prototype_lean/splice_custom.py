import torch
from transformers import CLIPModel, CLIPProcessor
import splice

class VLMBackbone:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs)  # [batch, 1024]

    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs)  # [num_texts, 1024]

def get_splice_model(image_mean_path="/mnt/ceph/storage/data-tmp/2025/uk077493/image_mean_flickr_2k_vit_h14.pt", device="cuda"):
    # Setup backbone
    vlm_backbone = VLMBackbone()

    # Build vocabulary
    concepts = splice.get_vocabulary("mscoco", 10000)
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