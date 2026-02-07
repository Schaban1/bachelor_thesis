import torch
import splice
from pathlib import Path
from constants import RESOURCES_DIR

class VLMBackbone(torch.nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe  # SD pipe für text encoder

    def encode_image(self, image):

        return image

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
        summary_token = text_embeds[:, -1:, :].squeeze(1)  # (B,768)
        return summary_token

def get_splice_model(pipe, device="cuda"):
    vlm_backbone = VLMBackbone(pipe)

    concepts_tensor = torch.load(RESOURCES_DIR / "concepts_tensor_laion_10k.pt", map_location="cpu").to(device)

    image_mean = torch.mean(concepts_tensor, dim=0)

    splicemodel = splice.SPLICE(image_mean, concepts_tensor, clip=vlm_backbone, device=device, return_weights=True)
    splicemodel.eval()
    return splicemodel