from concept_extractor import ConceptExtractor
from image_editor import ImageEditor

class SliderController:
    def __init__(self, webUI, splice_model, generator):
        self.webUI = webUI
        self.extractor = ConceptExtractor(splice_model)
        self.editor = ImageEditor(generator, splice_model)
        self.concept_maps = [{} for _ in range(webUI.num_images_to_generate)]
        self.offsets = [{} for _ in range(webUI.num_images_to_generate)]
        self.image_cache = {}

    def on_images_generated(self, images):
        self.webUI.update_image_displays()
        concepts_per_image = []

        for i, img in enumerate(images):
            print(f"[DEBUG slider_controller on_images_generated loop] Input img {i}: {type(img)}",flush=True)
            concepts = self.extractor.extract_top_concepts(img)  # [(name, idx), ...]
            self.concept_maps[i] = {name: idx for name, idx in concepts}
            self.offsets[i] = {idx: 0.0 for _, idx in concepts}

            # Store original image in cache
            zero_offsets = tuple(sorted(self.offsets[i].items()))
            cache_key = (i, zero_offsets)
            self.image_cache[cache_key] = img
            print(f"[CACHE] Stored original image {i} at {zero_offsets}")

            concepts_per_image.append([(name, 0.0) for name, _ in concepts])

        self.webUI.main_loop_ui.refresh_sliders(concepts_per_image)

    def on_slider_change(self, image_idx, concept_name, value):
        # Get concept index
        concept_idx = self.concept_maps[image_idx][concept_name]
        self.offsets[image_idx][concept_idx] = value

        current_offsets = tuple(sorted(self.offsets[image_idx].items()))
        cache_key = (image_idx, current_offsets)

        if cache_key in self.image_cache:
            new_img = self.image_cache[cache_key]
            self.webUI.main_loop_ui.on_image_cached(True)
        else:
            new_img = self.editor.edit_image(
                base_image=self.webUI.images[image_idx],
                concept_offsets=dict(current_offsets),
                image_idx=image_idx,
                loading_progress=self.webUI.loading_ui.loading_progress,
                queue_lock=self.webUI.queue_lock
            )
            self.image_cache[cache_key] = new_img
            self.webUI.main_loop_ui.on_image_cached(False)

        # Update
        self.webUI.images[image_idx] = new_img
        self.webUI.update_image_displays(single_idx=image_idx)