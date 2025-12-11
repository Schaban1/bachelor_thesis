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
        # self.sae_extractor = SAEExtractor(sae_model, concept_names)
        self.splice_extractor = SpliceExtractor(splice_model)
        self.editor = ImageEditor(generator, splice_model, sae_model)

        # State Containers
        self.concept_maps = [{} for _ in range(webUI.num_images_to_generate)]
        self.concept_maps_splice = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets_splice = [{} for _ in range(webUI.num_images_to_generate)]

        # Caches
        self.image_cache = {}
        self.image_cache_splice = {}
        self.original_images = {}

    def _get_cache_key(self, image_idx, offsets_dict):
        """
        Generates a deterministic, immutable cache key.
        Rounds floats to 4 decimals to avoid precision mismatches.
        """
        # Sort items by concept_idx (key) to ensure order doesn't matter
        sorted_items = sorted(offsets_dict.items())
        # Round values to ensure 5.0 and 5.0000001 are treated as the same state
        rounded_items = tuple((k, round(v, 4)) for k, v in sorted_items)
        return image_idx, rounded_items

    def load_fresh_sae(self):
        """Loads a new SAE model from disk"""
        print("[DEBUG] Loading FRESH SAE Model from disk...", flush=True)
        config = SparseAutoencoderConfig(
            n_input_features=1024,
            n_learned_features=8192,
        )
        clean_sae = SparseAutoencoder(config).to(self.generator.device)

        SAE_PATH = RESOURCES_DIR / "sparse_autoencoder_final.pt"
        state_dict = torch.load(SAE_PATH, map_location=self.generator.device)

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.endswith("_weight"):
                new_key = key.replace("_weight", "weight")
            elif key.endswith("_bias") and "encoder" in key:
                new_key = key.replace("_bias", "bias")
            if len(value.shape) > 0 and value.shape[0] == 1:
                value = value.squeeze(0)
            new_state_dict[new_key] = value

        print("Fixing dimensions in state_dict...")
        clean_sae.load_state_dict(new_state_dict, strict=False)
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

            # --- SAE SETUP ---
            concepts = self.sae_extractor.extract_top_concepts(img)  # [(name, idx), ...]
            self.concept_maps[i] = {name: idx for name, _, idx in concepts}
            # Initialize offsets to 0.0
            self.offsets[i] = {idx: 0.0 for _, _, idx in concepts}

            # Store Initial State in Cache
            cache_key = self._get_cache_key(i, self.offsets[i])
            self.image_cache[cache_key] = img.copy()  # Store copy to be safe
            print(f"[CACHE] SAE row – stored original image {i}")

            concepts_per_image.append([(name, value) for name, value, _ in concepts])

            # --- SPLICE SETUP ---
            splice_concepts = self.splice_extractor.extract_top_concepts(img)
            self.concept_maps_splice[i] = {name: idx for name, _, idx in splice_concepts}
            # Initialize offsets to 0.0
            self.offsets_splice[i] = {idx: 0.0 for _, _, idx in splice_concepts}

            # Store Initial State (All Zeros) in Cache
            cache_key_splice = self._get_cache_key(i, self.offsets_splice[i])
            self.image_cache_splice[cache_key_splice] = img.copy()
            print(f"[CACHE] SpLiCE row – stored original image {i}")

            splice_concepts_per_image.append([(name, value) for name, value, _ in splice_concepts])

        self.webUI.main_loop_ui.refresh_sliders(concepts_per_image, splice_concepts_per_image)

    def on_slider_change(self, image_idx, concept_name, value, is_sae=False):
        if is_sae:
            # 1. Update the state
            concept_idx = self.concept_maps[image_idx][concept_name]
            self.offsets[image_idx][concept_idx] = float(value)

            # 2. Generate Key from current state
            cache_key = self._get_cache_key(image_idx, self.offsets[image_idx])

            # 3. Cache Lookup
            if cache_key in self.image_cache:
                new_img = self.image_cache[cache_key].copy()
                self.webUI.main_loop_ui.on_image_cached(True)
            else:
                new_img = self.editor.sae_edit(
                    base_image=self.original_images[image_idx],
                    concept_offsets=self.offsets[image_idx],
                    image_idx=image_idx,
                    loading_progress=self.webUI.loading_ui.loading_progress,
                    queue_lock=self.webUI.queue_lock
                )
                # Store COPY in cache
                self.image_cache[cache_key] = new_img.copy()
                self.webUI.main_loop_ui.on_image_cached(False)

            # 4. Update UI
            self.webUI.images_sae[image_idx] = new_img
            self.webUI.update_image_displays(single_idx=image_idx)

        else:
            # Splice editing
            concept_idx = self.concept_maps_splice[image_idx][concept_name]
            self.offsets_splice[image_idx][concept_idx] = float(value)

            cache_key = self._get_cache_key(image_idx, self.offsets_splice[image_idx])

            if cache_key in self.image_cache_splice:
                new_img = self.image_cache_splice[cache_key].copy()
                self.webUI.main_loop_ui.on_image_cached(True)
            else:
                new_img = self.editor.splice_edit(
                    base_image=self.original_images[image_idx],
                    concept_offsets=self.offsets_splice[image_idx],
                    image_idx=image_idx,
                    loading_progress=self.webUI.loading_ui.loading_progress,
                    queue_lock=self.webUI.queue_lock
                )
                self.image_cache_splice[cache_key] = new_img.copy()
                self.webUI.main_loop_ui.on_image_cached(False)

            self.webUI.images_splice[image_idx] = new_img
            self.webUI.update_image_displays(single_idx=image_idx)