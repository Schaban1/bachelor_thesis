from nicegui import ui as ngUI
from constants import WebUIState
from .ui_component import UIComponent
import torch
import splice

class MainLoopUI(UIComponent):
    def build_userinterface(self):
        splicemodel = self.webUI.generator.splice
        vocabulary = splice.get_vocabulary("mscoco")
        with ngUI.column().classes('mx-auto items-center pl-24 pr-24').bind_visibility_from(self.webUI, 'is_main_loop_iteration', value=True):
            ngUI.label('Adjust sliders to edit images (one slider at a time).').style('font-size: 200%;')
            with ngUI.column().classes('mx-auto items-center'):
                with ngUI.row().classes('w-full items-center justify-start'):
                    ngUI.icon('subject', size='2rem').classes('mr-2')
                    ngUI.label(self.webUI.user_prompt).style('font-size: 120%;').bind_text_from(self.webUI, 'user_prompt')
            with ngUI.row().classes('mx-auto items-center mt-4'):
                self.sliders = [{} for _ in range(self.webUI.num_images_to_generate)]
                for i in range(self.webUI.num_images_to_generate):
                    with ngUI.column().classes('mx-auto items-center'):
                        self.webUI.images_display[i] = ngUI.interactive_image().style(f'width: {self.webUI.image_display_width}px; height: {self.webUI.image_display_height}px; object-fit: scale-down;')
                        preprocessed = self.webUI.generator.vlm_backbone.processor(images=self.webUI.images[i], return_tensors="pt").pixel_values.to(self.device)
                        sparse_weights = splicemodel.encode_image(preprocessed)  # [1, vocab_size] weights (concepts)
                        top_indices = torch.topk(sparse_weights[0], k=3).indices.tolist()  # Top 3 concepts
                        concept_names = [vocabulary[idx] for idx in top_indices]
                        with ngUI.column().classes('w-full'):
                            for concept, idx in zip(concept_names, top_indices):
                                slider = ngUI.slider(min=-0.3, max=0.3, step=0.15, value=0).props('label')
                                slider.on('update:model-value',lambda e, i=i, idx=idx: self.on_slider_change(i, idx, e['args']))
                                self.sliders[i][idx] = slider
            ngUI.space()

    def on_slider_change(self, image_idx, concept_idx, value):
        splicemodel = self.webUI.generator.splice  # Fix: Use correct reference
        for idx, slider in self.sliders[image_idx].items():
            if idx != concept_idx:
                slider.props('disable')
        # Get weights, adjust, recompose
        preprocessed = self.webUI.generator.vlm_backbone.processor(images=self.webUI.images[image_idx],
                                                                   return_tensors="pt").pixel_values.to(self.device)
        sparse_weights = splicemodel.encode_image(preprocessed)  # [1, vocab_size]
        sparse_weights[0, concept_idx] = max(0, min(1, sparse_weights[0, concept_idx] + value))  # Clamp [0, 1]
        recomposed_embedding = splicemodel.recompose_image(sparse_weights)  # [1, 1024]
        self.webUI.images[image_idx] = self.webUI.generator.generate_with_splice(
            self.webUI.images[image_idx], recomposed_embedding, self.webUI.loading_ui.loading_progress,
            self.webUI.queue_lock
        )[0]
        self.webUI.update_image_displays()
        for slider in self.sliders[image_idx].values():
            slider.props('enable')