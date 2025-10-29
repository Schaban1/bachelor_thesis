import torch
import splice

class ConceptExtractor:
    def __init__(self, splice_model):
        self.splice = splice_model
        self.vocabulary = splice.get_vocabulary("mscoco", 10000)

    def extract_top_concepts(self, pil_image, topk=3):
        inputs = self.splice.clip.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.splice.clip.device)

        sparse_weights = self.splice.encode_image(pixel_values)
        topk_indices = torch.topk(sparse_weights[0], k=topk).indices.tolist()

        concepts = [self.vocabulary[i] for i in topk_indices]
        return list(zip(concepts, topk_indices))