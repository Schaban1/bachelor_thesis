from concept_extractor import SpliceExtractor, SAEExtractor
from image_editor import ImageEditor

class SliderController:
    def __init__(self, webUI, splice_model, generator, sae_model, concept_names):
        self.webUI = webUI
        self.sae_extractor = SAEExtractor(sae_model, concept_names)
        self.splice_extractor = SpliceExtractor(splice_model)
        self.editor = ImageEditor(generator, splice_model, sae_model)
        self.concept_maps = [{} for _ in range(webUI.num_images_to_generate)]
        self.concept_maps_splice = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets_splice = [{} for _ in range(webUI.num_images_to_generate)]
        self.image_cache = {}
        self.image_cache_splice = {}

    def on_images_generated(self, images):
        self.webUI.update_image_displays()
        concepts_per_image = []
        splice_concepts_per_image = []

        for i, img in enumerate(images):
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
            cache_key = (image_idx, tuple(current_offsets.items()))
            if cache_key in self.image_cache:
                new_img = self.image_cache[cache_key]
                self.webUI.main_loop_ui.on_image_cached(True)
            else:
                new_img = self.editor.sae_edit(
                    base_image=self.webUI.images[image_idx],
                    concept_offsets=current_offsets,
                    image_idx=image_idx,
                    loading_progress=self.webUI.loading_ui.loading_progress,
                    queue_lock=self.webUI.queue_lock
                )
                self.image_cache[cache_key] = new_img
                self.webUI.main_loop_ui.on_image_cached(False)
            self.webUI.images[image_idx] = new_img
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
                    base_image=self.webUI.images[image_idx],
                    concept_offsets=current_offsets,
                    image_idx=image_idx,
                    loading_progress=self.webUI.loading_ui.loading_progress,
                    queue_lock=self.webUI.queue_lock
                )
                self.image_cache_splice[cache_key] = new_img
                self.webUI.main_loop_ui.on_image_cached(False)
            self.webUI.images[image_idx] = new_img
            self.webUI.update_image_displays(single_idx=image_idx)
