import torch
import splice

class ConceptExtractor:
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

        print(f"[DEBUG] concepts → type: {type(concepts)} | length: {len(concepts)}", flush=True)
        print(f"[DEBUG] concepts → {concepts}", flush=True)
        print(f"[DEBUG] values → {topk_values}", flush=True)
        print(f"[DEBUG] final return → {list(zip(concepts, topk_indices))}", flush=True)
        print("[DEBUG conceptsextractor: were the concepts extracted?]?", flush=True)
        return list(zip(concepts, topk_values, topk_indices))