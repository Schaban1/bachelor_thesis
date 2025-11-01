# generator/image_editor.py
from collections import defaultdict

class ImageEditor:
    def __init__(self, generator, splice_model):
        self.generator = generator
        self.splice = splice_model
        self.cache = defaultdict(dict)  # (img_idx, state_key) -> PIL image

    def edit_image(self, base_image, concept_offsets, image_idx, loading_progress=None, queue_lock=None):
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
        )[0]

        # Cache
        self.cache[image_idx][state_key] = new_img
        return new_img