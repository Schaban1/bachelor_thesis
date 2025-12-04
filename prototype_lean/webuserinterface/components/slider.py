from concept_extractor import SpliceExtractor, SAEExtractor
from image_editor import ImageEditor
from sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
from constants import RESOURCES_DIR
import torch

class SliderController:
    def __init__(self, webUI, splice_model, generator, sae_model, concept_names):
        self.webUI = webUI
        self.sae_model = sae_model
        self.concept_names = concept_names
        self.generator = generator
        #self.sae_extractor = SAEExtractor(sae_model, concept_names)
        self.splice_extractor = SpliceExtractor(splice_model)
        self.editor = ImageEditor(generator, splice_model, sae_model)
        self.concept_maps = [{} for _ in range(webUI.num_images_to_generate)]
        self.concept_maps_splice = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets_splice = [{} for _ in range(webUI.num_images_to_generate)]
        self.image_cache = {}
        self.image_cache_splice = {}
        self.original_images = {}

    def load_fresh_sae(self):
        """Loads a new SAE model from disk"""
        print("[DEBUG] Loading FRESH SAE Model from disk...", flush=True)
        config = SparseAutoencoderConfig(
            n_input_features=1024,
            n_learned_features=8192,
        )
        # Create new instance
        clean_sae = SparseAutoencoder(config).to(self.generator.device)


        SAE_PATH = RESOURCES_DIR / "sparse_autoencoder_final.pt"
        state_dict = torch.load(SAE_PATH, map_location=self.generator.device)

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key

            # 1. Fix: encoder._weight -> encoder.weight
            if key.endswith("_weight"):
                new_key = key.replace("_weight", "weight")

            # 2. Fix: encoder._bias -> encoder.bias
            elif key.endswith("_bias") and "encoder" in key:
                new_key = key.replace("_bias", "bias")

            # 3. Squeeze Check
            if len(value.shape) > 0 and value.shape[0] == 1:
                value = value.squeeze(0)

            new_state_dict[new_key] = value

        print("\n=== MODEL EXPECTATION VS CHECKPOINT REALITY ===")

        # 1. What the Model wants (The Empty Slots)
        model_keys = set(clean_sae.state_dict().keys())
        print(f"Model expects these keys: {sorted(list(model_keys))}")

        # 2. What the File has (The Data)

        file_keys = set(new_state_dict.keys())
        print(f"File contains these keys: {sorted(list(file_keys))}")

        # 3. The Mismatch
        print(f"Missing in File (Model won't load these): {model_keys - file_keys}")
        print(f"Extra in File (Model will ignore these): {file_keys - model_keys}")
        print("===============================================\n")

        keys = clean_sae.load_state_dict(new_state_dict, strict=False)
        print(f"[DEBUG] Missing Keys: {keys.missing_keys}")

        clean_sae.eval()
        return clean_sae

    def on_images_generated(self, images):
        clean_sae_model = self.load_fresh_sae()
        self.sae_extractor = SAEExtractor(clean_sae_model, self.concept_names)
        concepts_per_image = []
        splice_concepts_per_image = []

        # Clear old caches for new generation
        self.image_cache = {}
        self.image_cache_splice = {}
        self.original_images = {}

        for i, img in enumerate(images):
            self.original_images[i] = img.copy()
            # SAE ROW
            concepts = self.sae_extractor.extract_top_concepts(img)  # [(name, idx), ...]
            self.concept_maps[i] = {name: idx for name, _, idx in concepts}
            self.offsets[i] = {idx: 0.0 for _, _, idx in concepts}
            # Store original image in cache
            zero_offsets = tuple(sorted(self.offsets[i].items()))
            cache_key = (i, zero_offsets)
            self.image_cache[cache_key] = img
            print(f"[CACHE] SAE row – stored original image {i}")

            concepts_per_image.append([(name, value) for name, value, _ in concepts])

            # SPLICE ROW
            splice_concepts = self.splice_extractor.extract_top_concepts(img)
            self.concept_maps_splice[i] = {name: idx for name, _, idx in splice_concepts}
            self.offsets_splice[i] = {idx: 0.0 for _, _, idx in splice_concepts}

            zero_offsets_splice = tuple(sorted(self.offsets_splice[i].items()))
            cache_key_splice = (i, zero_offsets_splice)
            self.image_cache_splice[cache_key_splice] = img
            print(f"[CACHE] SpLiCE row – stored original image {i}")

            splice_concepts_per_image.append([(name, value) for name, value, _ in splice_concepts])

        self.webUI.main_loop_ui.refresh_sliders(concepts_per_image, splice_concepts_per_image)

    def on_slider_change(self, image_idx, concept_name, value, is_sae=False):
        if is_sae:
            concept_idx = self.concept_maps[image_idx][concept_name]
            self.offsets[image_idx][concept_idx] = value
            current_offsets = dict(sorted(self.offsets[image_idx].items()))

            # Use separate cache for SAE
            cache_key = (image_idx, tuple(current_offsets.items()))
            if cache_key in self.image_cache:
                new_img = self.image_cache[cache_key]
                self.webUI.main_loop_ui.on_image_cached(True)
            else:
                new_img = self.editor.sae_edit(
                    base_image=self.original_images[image_idx],
                    concept_offsets=current_offsets,
                    image_idx=image_idx,
                    loading_progress=self.webUI.loading_ui.loading_progress,
                    queue_lock=self.webUI.queue_lock
                )
                self.image_cache[cache_key] = new_img
                self.webUI.main_loop_ui.on_image_cached(False)
            self.webUI.images_sae[image_idx] = new_img
            self.webUI.update_image_displays(single_idx=image_idx)

        else:
            # Splice editing
            concept_idx = self.concept_maps_splice[image_idx][concept_name]
            self.offsets_splice[image_idx][concept_idx] = value
            current_offsets = dict(sorted(self.offsets_splice[image_idx].items()))

            cache_key = (image_idx, tuple(current_offsets.items()))

            if cache_key in self.image_cache_splice:
                new_img = self.image_cache_splice[cache_key]
                self.webUI.main_loop_ui.on_image_cached(True)
            else:
                new_img = self.editor.splice_edit(
                    base_image=self.original_images[image_idx],
                    concept_offsets=current_offsets,
                    image_idx=image_idx,
                    loading_progress=self.webUI.loading_ui.loading_progress,
                    queue_lock=self.webUI.queue_lock
                )
                self.image_cache_splice[cache_key] = new_img
                self.webUI.main_loop_ui.on_image_cached(False)
            self.webUI.images_splice[image_idx] = new_img
            self.webUI.update_image_displays(single_idx=image_idx)
