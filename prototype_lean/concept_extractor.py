import torch
import splice

class ConceptExtractor:
    def __init__(self, splice_model):
        self.splice = splice_model
        self.vocabulary = splice.get_vocabulary("mscoco", 10000)

    def extract_top_concepts(self, pil_image, topk=3):
        '''
        print(f"[DEBUG extract_top_concepts] Input pil_image: {type(pil_image)}")
        inputs = self.splice.clip.processor(images=pil_image, return_tensors="pt")
        print(f"[DEBUG extract_top_concepts] Input inputs: {type(inputs)}")
        pixel_values = inputs.pixel_values.to(self.splice.clip.device)
        print(f"[DEBUG extract_top_concepts] Input pixel_values: {type(pixel_values)}")
        '''

        sparse_weights = self.splice.encode_image(pil_image)
        topk_indices = torch.topk(sparse_weights[0], k=topk).indices.tolist()

        concepts = [self.vocabulary[i] for i in topk_indices]
        print("[DEBUG conceptsextractor: were the concepts extracted?]?", flush=True)
        return list(zip(concepts, topk_indices))