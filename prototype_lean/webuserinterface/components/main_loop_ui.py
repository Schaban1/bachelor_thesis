from nicegui import ui as ngUI
from .ui_component import UIComponent


class MainLoopUI(UIComponent):
    def create_label_updater(self, name, base_value, is_relative=False):
        """
        Creates a dedicated function for updating labels.
        - If is_relative=True (SAE): Calculates Base * (1 + v)
        - If is_relative=False (Splice): Calculates Base + v
        """
        if is_relative:
            # SAE Logic: Percentage change (e.g. 0.05 is +5%)
            return lambda v: f"{name}: {max(0.0, base_value + v):.2f}"
        else:
            # Splice Logic: Absolute addition (e.g. 0.1 is +0.1)
            return lambda v: f"{name}: {max(0.0, base_value + v):.2f}"
    def build_userinterface(self):
        print("[DEBUG] build_userinterface: START", flush=True)
        with ngUI.column().classes('mx-auto items-center pl-24 pr-24') \
                .bind_visibility_from(self.webUI, 'is_main_loop_iteration', value=True):
            ngUI.label('Edit images by adjusting concept sliders.').style('font-size: 200%;')

            with ngUI.row().classes('w-full items-center justify-start'):
                ngUI.icon('subject', size='2rem').classes('mr-2')
                ngUI.label().bind_text_from(self.webUI, 'user_prompt').style('font-size: 120%;')

            # FIRST ROW: SAE extractor
            with ngUI.row().classes('mx-auto items-start mt-4 gap-8 justify-center'):
                with ngUI.column().classes('w-full items-center'):
                    ngUI.label('SAE Extractor').classes('text-center font-bold')
                    ngUI.label(
                        "SAE sliders adjust concept strength relatively in 5 % steps (min./max.: ±10%)"
                    ).classes('text-sm text-gray-600 italic mb-2 text-center')
                for i in range(self.webUI.num_images_to_generate):
                    with ngUI.column().classes('items-center'):
                        self.webUI.images_display[i] = ngUI.interactive_image() \
                            .style(f'width: {self.webUI.image_display_width}px; '
                                    f'height: {self.webUI.image_display_height}px; '
                                    f'object-fit: scale-down; border: 2px solid #ccc;')
                        container = ngUI.column().classes('w-full mt-2 space-y-1')
                        self.webUI.slider_containers.append(container)  # SAE sliders

            # SECOND ROW: Splice extractor
            with ngUI.row().classes('mx-auto items-start mt-4 gap-8 justify-center'):
                with ngUI.column().classes('w-full items-center'):
                    ngUI.label('Splice Extractor').classes('text-center font-bold')
                    ngUI.label(
                        "Splice sliders adjust concept strength in absolute 0.1 steps (min./max.: ±0.2)"
                    ).classes('text-sm text-gray-600 italic mb-2 text-center')
                for i in range(self.webUI.num_images_to_generate):
                    with ngUI.column().classes('items-center'):
                        self.webUI.images_display_splice[i] = ngUI.interactive_image() \
                            .style(f'width: {self.webUI.image_display_width}px; '
                                   f'height: {self.webUI.image_display_height}px; '
                                   f'object-fit: scale-down; border: 2px solid #ccc;')

                        # One container per image
                        container = ngUI.column().classes('w-full mt-2 space-y-1')
                        self.webUI.slider_containers_splice.append(container)

            ngUI.space()
            print("[DEBUG mainloop webuserinterface builduserinterface: was async def build_userinterface() called?",flush=True)

    def refresh_sliders(self, concepts_per_image, splice_concepts_per_image):
        # SAE (first row)
        print("[DEBUG] refresh_sliders called with {len(concepts_per_image)} images",flush=True)
        for idx, container in enumerate(self.webUI.slider_containers):
            container.clear()
            print("[DEBUG] Clearing container {idx}",flush=True)
            with container:
                for concept_name, concept_value in concepts_per_image[idx]:
                    # CONCEPT NAME
                    ngUI.label(concept_name).classes('text-center font-bold text-sm mb-1 text-blue-600')
                    # SLIDER ROW
                    with ngUI.row().classes('w-full items-center gap-2'):
                        # LEFT: Less
                        ngUI.label("Less").classes('text-xs text-gray-500 w-12 text-left')
                        # MIDDLE: Slider + Value
                        with ngUI.row().classes('flex-grow items-center'):
                            # FIX: Added 'label-value' prop to show % sign on the handle
                            slider = ngUI.slider(min=-0.1, max=0.1, step=0.05, value=0) \
                                .props('label-always') \
                                .classes('flex-grow')
                            ngUI.label().bind_text_from(
                                slider, 'value',
                                backward=self.create_label_updater(concept_name, concept_value, is_relative=True)
                            ).classes('text-xs text-gray-500 w-32 text-center')
                        # RIGHT
                        ngUI.label("More").classes('text-xs text-gray-500 w-12 text-right')
                        slider.on('update:model-value',
                                  lambda e, i=idx, c=concept_name:
                                  self.webUI.slider_controller.on_slider_change(i, c, e.args, is_sae=True)
                                  )
        # SpLiCE (second row)
        for idx, container in enumerate(self.webUI.slider_containers_splice):
            container.clear()
            with container:
                for concept_name, value in splice_concepts_per_image[idx]:
                    ngUI.label(concept_name).classes('text-center font-bold text-sm mb-1 text-blue-600')
                    with ngUI.row().classes('w-full items-center gap-2'):
                        ngUI.label("Less").classes('text-xs text-gray-500 w-12 text-left')
                        with ngUI.row().classes('flex-grow items-center'):
                            slider = ngUI.slider(min=-0.2, max=0.2, step=0.1, value=0) \
                                .props('label-always') \
                                .classes('flex-grow')
                            ngUI.label().bind_text_from(
                                slider, 'value',
                                backward=self.create_label_updater(concept_name, value)
                            ).classes('text-xs text-gray-500 w-32 text-center')
                        ngUI.label("More").classes('text-xs text-gray-500 w-12 text-right')
                        slider.on('update:model-value',
                                lambda e, i=idx, c=concept_name:
                                self.webUI.slider_controller.on_slider_change(i, c, e.args, is_sae=False)
                                )


    def on_image_cached(self, was_cached):
        if was_cached:
            ngUI.notify("↺ Cached image loaded!", type='positive')
        else:
            ngUI.notify("✨ New image generated!",type='positive')