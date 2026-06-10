import torch
import splice
from pathlib import Path
from constants import RESOURCES_DIR
from transformers import CLIPModel, CLIPProcessor

class VLMBackbone(torch.nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe  # SD pipe für text encoder
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs)
        #return image

    def encode_text(self, text):

        text_inputs = self.pipe.tokenizer(
            text,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.pipe.device)
        with torch.no_grad():
            text_embeds = self.pipe.text_encoder(text_inputs.input_ids)[0]
        summary_token = text_embeds[:, text_inputs.attention_mask.sum() - 2, :]
        return summary_token

def get_splice_model(pipe, device="cuda"):
    vlm_backbone = VLMBackbone(pipe)

    #concepts_tensor = torch.load(RESOURCES_DIR / "concepts_tensor_laion_10k.pt", map_location="cpu").to(device)
    #image_mean = torch.mean(concepts_tensor, dim=0)

    concepts = splice.get_vocabulary("laion", 10000)
    embedded_concepts = []
    for concept in concepts:
        emb = vlm_backbone.encode_text(concept)
        embedded_concepts.append(emb.squeeze(0))

    concepts_tensor = torch.stack(embedded_concepts).float()

    concepts_tensor = torch.nn.functional.normalize(concepts_tensor, dim=1)

    image_mean = torch.mean(concepts_tensor, dim=0)

    concepts_tensor = concepts_tensor - torch.mean(concepts_tensor, dim=0)
    concepts_tensor = torch.nn.functional.normalize(concepts_tensor, dim=1)

    splicemodel = splice.SPLICE(image_mean, concepts_tensor, clip=vlm_backbone, device=device, return_weights=True)
    splicemodel.eval()
    return splicemodel